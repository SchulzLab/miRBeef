import os
import numpy as np
import torch
import os

# Check the current conda environment
conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')

# Conditional import based on the conda environment
if conda_env == 'mitar':
    import pytorch_lightning as L
    print("Using 'pytorch_lightning' for the 'mitar' environment.")
else:
    import lightning as L
    print("Using 'lightning' for the environment:", conda_env)

from captum.attr import IntegratedGradients, LayerIntegratedGradients
from matplotlib import pyplot as plt
from torch import nn, optim
from scipy import stats
from sklearn.metrics import confusion_matrix, average_precision_score, roc_auc_score, matthews_corrcoef, \
    precision_score, recall_score, f1_score, brier_score_loss

from .models import DMISO, CNNSequenceModel, CNNTransformerModel, HybridSequenceModel, MitarNet, TransPHLA, TECMiTarget, GraphTar
from ..utils import plot_weights

class MiTarModule(L.LightningModule):
    def __init__(
            self,
            model_name="CNNSequenceModel",
            model_config=None,
            opt_name="Adam",
            opt_config=None,
            pos_weight=None,  # Changed here #####################
    ):
        super().__init__()
        # Exports the hyperparameters to a YAML file
        self.save_hyperparameters()
        if model_config is None:
            model_config = {}
        self.opt_name = opt_name
        if opt_config is None:
            opt_config = {}
        self.opt_config = opt_config
        # create model
        if model_name == "HybridSequenceModel":
            self.model = HybridSequenceModel(**model_config)
        elif model_name == "CNNSequenceModel":
            self.model = CNNSequenceModel(**model_config)
        elif model_name == "CNNTransformerModel":
            self.model = CNNTransformerModel(**model_config)
        elif model_name == "DMISO":
            self.model = DMISO(**model_config)
        elif model_name == "MiTar":
            self.model = MitarNet(**model_config)
        elif model_name == "TransPHLA":
            self.model = TransPHLA(**model_config)
        elif model_name == "TEC-miTarget":
            self.model = TECMiTarget(**model_config)
            self.pos_weight = pos_weight # Changed here #####################
        elif model_name == "GraphTar":
            self.model = GraphTar(**model_config)
            
         # Changed here #####################
        if self.pos_weight is not None:
            self.loss_module = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        else:
            self.loss_module = nn.BCELoss()
         # Changed here #####################
        
        self.training_step_outputs = []
        self.test_step_outputs = []

    def configure_optimizers(self):
        optimizer_dict = {
            "Adam": optim.AdamW,
            "SGD": optim.SGD,
            "Adadelta": optim.Adadelta
        }
        if self.opt_name not in optimizer_dict:
            raise ValueError(f"Unsupported optimizer: {self.opt_name}. Supported optimizers are {list(optimizer_dict.keys())}")
        optimizer = optimizer_dict[self.opt_name](self.parameters(), **self.opt_config)

        # reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=4)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"  # Validation metric to monitor
            }
        }

    def basic_step(self, batch):
        if isinstance(self.model, GraphTar):
            preds = self.model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch)
            ys = batch.y
        else:
            if isinstance(self.model, CNNSequenceModel) and self.model.incorporate_type:
                preds = self.model(mirna_input=batch[0], target_input=batch[1], type_input=batch[2])
            else:
                preds = self.model(mirna_input=batch[0], target_input=batch[1])
            ys = batch[-1].float()
        preds = preds.squeeze(dim=1)
        loss = self.loss_module(preds, ys)
        pred_labels = (preds >= 0.5).long()
        acc = (ys == pred_labels).float().mean()
        return loss, acc, pred_labels

    def training_step(self, batch, batch_idx):
        loss, acc, pred_labels = self.basic_step(batch)

        # the 'loss' key is mandatory, it tells PTL what to optimize
        training_step_info = {
            'loss': loss,
            'train_acc': acc,
            'targets': batch.y if isinstance(self.model, GraphTar) else batch[-1],
            'predictions': pred_labels
        }
        self.training_step_outputs.append(training_step_info)

        # logs the accuracy per epoch (weighted average over batches)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # free memory
        torch.cuda.empty_cache()
        return training_step_info

    def on_train_epoch_end(self):
        all_targets = []
        all_predictions = []

        for output in self.training_step_outputs:
            all_targets += output['targets']  # labels
            all_predictions += output['predictions']

        # Convert lists to tensors
        all_targets = torch.tensor(all_targets)
        all_predictions = torch.tensor(all_predictions)

        # Calculate confusion matrix
        conf_matrix = confusion_matrix(all_targets, all_predictions)
        if conf_matrix.shape == (1, 1):
            conf_matrix = np.array([[conf_matrix[0, 0], 0], [0, 0]])
        true_positives = conf_matrix[1, 1]
        false_positives = conf_matrix[0, 1]
        true_negatives = conf_matrix[0, 0]
        false_negatives = conf_matrix[1, 0]

        recall = true_positives / (true_positives + false_negatives + 1e-8)  # Avoid division by zero
        precision = true_positives / (true_positives + false_positives + 1e-8)

        print(
            f"\nEpoch {self.current_epoch}: TP: {true_positives}, FP: {false_positives}, TN: {true_negatives}, FN: {false_negatives}, Precision: {precision:.2f}, Recall: {recall:.2f}")

        self.training_step_outputs.clear()  # free memory

    def validation_step(self, batch, batch_idx):
        loss, acc, pred_labels = self.basic_step(batch)
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        # free memory
        torch.cuda.empty_cache()

    def test_step(self, batch, batch_idx):
        test_step_info = {"y": batch.y if isinstance(self.model, GraphTar) else batch[-1]}
        if not isinstance(self.model, GraphTar) and len(batch) > 3:
            categories = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]]).to(self.device)
            test_step_info["category"] = \
                torch.stack([(batch[2] == category).all(dim=1) for category in categories]).T.nonzero(as_tuple=True)[1]

        # get predictions
        if isinstance(self.model, GraphTar):
            preds = self.model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch)
        elif isinstance(self.model, CNNSequenceModel) and self.model.incorporate_type:
            preds = self.model(mirna_input=batch[0], target_input=batch[1], type_input=batch[2])
        else:
            preds = self.model(mirna_input=batch[0], target_input=batch[1])
        preds = preds.squeeze(dim=1)
        test_step_info["preds"] = preds

        # calculate average input attributions
        if isinstance(self.model, (CNNSequenceModel, DMISO, MitarNet)):
            test_step_info["avg_attrs_mirna"], test_step_info["avg_attrs_target"] = \
                self._calculate_average_attributions(batch)

        # calculate average input
        if not isinstance(self.model, GraphTar):
            test_step_info["mirna_differences"] = self._calculate_average_perturbation(batch, preds, 0)
            test_step_info["target_differences"] = self._calculate_average_perturbation(batch, preds, 1)

        # save tmp information
        self.test_step_outputs.append(test_step_info)

        # free memory
        torch.cuda.empty_cache()

        return test_step_info

    def compute_pos_weight(train_dataset): # Changed here #########################
        all_labels = torch.tensor([sample[-1] for sample in train_dataset])
        n_pos = (all_labels == 1).sum().float()
        n_neg = (all_labels == 0).sum().float()
        pos_weight = n_neg / (n_pos + 1e-8)
        self.log("pos_weight", pos_weight)
        return pos_weight

    def _calculate_average_perturbation(self, batch, preds, col_idx):
        differences = []
        for i in range(batch[col_idx].size(1)):
            # permute the i-th position of the sequence with n-encoding
            perm = batch[col_idx].clone().to(self.device)
            if perm.dim() == 2:
                perm[:, i] = self.model.n_encoding
            else:
                perm[:, i, :] = torch.tensor(self.model.n_encoding, dtype=torch.float32).to(self.device)

            # get predictions for permuted sequence
            if col_idx == 0:
                perm_preds = self.model(mirna_input=perm, target_input=batch[1])
            else:
                perm_preds = self.model(mirna_input=batch[0], target_input=perm)

            # calculate difference between original and permuted predictions
            perm_preds = perm_preds.squeeze(dim=1)
            differences.append(perm_preds - preds)

        return torch.stack(differences, dim=1).detach().cpu()

    '''def _calculate_average_attributions(self, batch):
        self.train()
        torch.set_grad_enabled(True)
        ig = IntegratedGradients(self.model)

        # Get attributions with fewer steps
        if isinstance(self.model, CNNSequenceModel) and self.model.incorporate_type:
            attributions = ig.attribute(inputs=(batch[0], batch[1], batch[2]), n_steps=5)
        elif isinstance(self.model, TransPHLA):
            attributions = []
            lig = LayerIntegratedGradients(self.model, self.model.mirna_embedding)
            attributions.append(lig.attribute(inputs=(batch[0], batch[1]), 
                                             baselines=(torch.full_like(batch[0], 4).long(), 
                                                        torch.full_like(batch[1], 4).long()), 
                                             n_steps=5))
            lig = LayerIntegratedGradients(self.model, self.model.target_embedding)
            attributions.append(lig.attribute(inputs=(batch[0], batch[1]), 
                                             baselines=(torch.full_like(batch[0], 4).long(), 
                                                        torch.full_like(batch[1], 4).long()), 
                                             n_steps=5))
        else:
            attributions = ig.attribute(inputs=(batch[0], batch[1]), n_steps=5)

        self.eval()
        avg_attrs_mirna = attributions[0].permute(0, 2, 1).detach().cpu().mean(dim=0)
        avg_attrs_target = attributions[1].permute(0, 2, 1).detach().cpu().mean(dim=0)
        
        return avg_attrs_mirna, avg_attrs_target
    '''
    def _calculate_average_attributions(self, batch):
        self.train()
        torch.set_grad_enabled(True)
        ig = IntegratedGradients(self.model)

        # Ensure inputs require gradients.
        input0 = batch[0].clone().detach().requires_grad_(True)
        input1 = batch[1].clone().detach().requires_grad_(True)

        # If your model incorporates a third input (e.g. type_input), enable grad for it as well.
        if hasattr(self.model, "incorporate_type") and self.model.incorporate_type:
            input2 = batch[2].clone().detach().requires_grad_(True)
            attributions = ig.attribute(inputs=(input0, input1, input2), n_steps=15)
        else:
            attributions = ig.attribute(inputs=(input0, input1), n_steps=15)

        self.eval()
        avg_attrs_mirna = attributions[0].permute(0, 2, 1).detach().cpu().mean(dim=0)
        avg_attrs_target = attributions[1].permute(0, 2, 1).detach().cpu().mean(dim=0)
        return avg_attrs_mirna, avg_attrs_target

    def on_test_epoch_end(self):
        self._log_confusion_matrices()
        os.makedirs(os.path.join(self.trainer.default_root_dir, "figures"), exist_ok=True)
        # self._log_heatmap()
        if isinstance(self.model, (CNNSequenceModel, DMISO, MitarNet)):
            self._log_attributions()
        # TODO: implement for other models
        if not isinstance(self.model, GraphTar):
            self._log_single_position_perturbation_ci()
        self.test_step_outputs.clear()

    def _log_confusion_matrices(self):
        y = torch.cat([out["y"] for out in self.test_step_outputs]).detach().cpu()
        preds = torch.cat([out["preds"] for out in self.test_step_outputs]).detach().cpu()

        # binary predictions
        pred_labels = (preds >= 0.5).long()

        # calculate metrics
        self.log("test_acc", (y == pred_labels).float().mean())
        self.log("test_precision", precision_score(y, pred_labels))
        self.log("test_recall", recall_score(y, pred_labels))
        self.log("test_f1", f1_score(y, pred_labels))
        self.log("test_mcc", matthews_corrcoef(y, pred_labels))
        self.log("test_auroc", roc_auc_score(y, preds))
        self.log("test_auprc", average_precision_score(y, preds))
        self.log("test_brier", brier_score_loss(y, preds))

        # log confusion matrices
        y = y.numpy()
        pred_labels = pred_labels.numpy()
        if hasattr(self.logger.experiment, "log_confusion_matrix"):
            self.logger.experiment.log_confusion_matrix(y, pred_labels, title="Test Confusion Matrix")
        else:
            print("Confusion matrix logging not supported by this logger.")
        category_dict = {0: "canonical", 1: "3p-isomir", 2: "5p-isomir", 3: "3p5p-isomir"}
        if "category" in self.test_step_outputs[0]:
            categories = torch.cat([out["category"] for out in self.test_step_outputs]).detach().cpu().numpy()
            for category in np.unique(categories):
                category_indices = categories == category
                self.logger.experiment.log_confusion_matrix(
                    y[category_indices], pred_labels[category_indices],
                    title=f"Test Confusion Matrix ({category_dict[category]})",
                    file_name=f"confusion_matrix_{category_dict[category]}.json"
                )

    def _log_attributions(self):
        # average attributions
        avg_attrs_mirna = torch.stack([out["avg_attrs_mirna"] for out in self.test_step_outputs]).mean(dim=0).numpy()
        avg_attrs_target = torch.stack([out["avg_attrs_target"] for out in self.test_step_outputs]).mean(dim=0).numpy()

        # calculate sum of weights for each position
        spacer_len = 1
        spacer = np.ones((self.model.target_dim, spacer_len)) * np.nan
        padded_avg_attrs_mirna = np.zeros((self.model.target_dim, self.model.max_mirna_len))
        padded_avg_attrs_mirna[:self.model.mirna_dim, :] = avg_attrs_mirna
        weights = np.concatenate([padded_avg_attrs_mirna, spacer, avg_attrs_target], axis=1)
        weights = np.abs(weights).sum(axis=0)
        normed_weights = weights / np.max(weights[~np.isnan(weights)])

        # plot weights
        fig_weights, ax = plt.subplots(figsize=(weights.shape[0], 15))
        ax.bar(range(weights.shape[0]), normed_weights)
        ax.set_xticks(range(weights.shape[0]))
        xtick_labels = (
            [f"M{i + 1}" for i in range(self.model.max_mirna_len)]
            + [""] * spacer_len
            + [f"T{i + 1}" for i in range(self.model.max_target_len)]
        )
        ax.set_xticklabels(xtick_labels, fontsize=20)
        path = os.path.join(self.trainer.default_root_dir, "figures", "average_attributions.png")
        fig_weights.savefig(path, bbox_inches='tight')
        # Only log if the logger supports log_image:
        if hasattr(self.logger.experiment, "log_image"):
            self.logger.experiment.log_image(path)
        else:
            print(f"Image logging not supported. Saved to {path}")
        plt.close(fig_weights)

    def _log_heatmap(self):
        # average attributions
        avg_attrs_mirna = torch.stack([out["avg_attrs_mirna"] for out in self.test_step_outputs]).mean(dim=0).numpy()
        avg_attrs_target = torch.stack([out["avg_attrs_target"] for out in self.test_step_outputs]).mean(dim=0).numpy()

        # create heatmap
        spacer_len = 1
        spacer = np.ones((len(self.model.n_encoding), spacer_len)) * np.nan
        heatmap = np.concatenate([avg_attrs_mirna, spacer, avg_attrs_target], axis=1)

        # plot heatmap
        fig_heatmap, ax = plt.subplots(figsize=(heatmap.shape[1], heatmap.shape[0] * 3))
        ax.imshow(heatmap, cmap='viridis')
        xtick_labels = (
            [f"M{i+1}" for i in range(self.model.max_mirna_len)]
            + [""] * spacer_len
            + [f"T{i+1}" for i in range(self.model.max_target_len)]
        )
        ax.set_xticks(np.arange(heatmap.shape[1]))
        ax.set_xticklabels(xtick_labels, fontsize=20)
        ax.set_yticks(np.arange(heatmap.shape[0]))
        if len(self.model.n_encoding) == 4:
            ax.set_yticklabels(["A", "T", "C", "G"], fontsize=20)
        else:
            ax.set_yticklabels(["N", "A", "T", "C", "G"], fontsize=20)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        path = os.path.join(self.trainer.default_root_dir, "figures", "heatmap.png")
        fig_heatmap.savefig(path, bbox_inches='tight')
        self.logger.experiment.log_image(path)
        plt.close(fig_heatmap)

    def _log_single_position_perturbation_ci(self):
        mirna_differences = torch.cat([out["mirna_differences"] for out in self.test_step_outputs], dim=0).detach()
        target_differences = torch.cat([out["target_differences"] for out in self.test_step_outputs], dim=0).detach()

        y = torch.cat([out["y"] for out in self.test_step_outputs]).detach().cpu()
        preds = torch.cat([out["preds"] for out in self.test_step_outputs]).detach().cpu()
        pred_labels = (preds >= 0.5).long()

        category_dict = {0: "canonical", 1: "3p-isomir", 2: "5p-isomir", 3: "3p5p-isomir"}
        if "category" in self.test_step_outputs[0]:
            categories = torch.cat([out["category"] for out in self.test_step_outputs]).detach().cpu().numpy()
            perturbation_plots = []
            for category in np.unique(categories):
                tp = (y == 1) & (pred_labels == 1) & (categories == category)
                pert_plot, ci_up, ci_low = self._single_position_perturbation_ci(
                    mirna_differences[tp], target_differences[tp], f"{category_dict[category]}-TP")
                perturbation_plots.append((category_dict[category], pert_plot, ci_up, ci_low))

                tn = (y == 0) & (pred_labels == 0) & (categories == category)
                _, _, _ = self._single_position_perturbation_ci(
                    mirna_differences[tn], target_differences[tn], f"{category_dict[category]}-TN")
                # TODO: log perturbation plot for TN if needed
            # log perturbation plots
            self._merge_perturbation_plots(perturbation_plots, mirna_differences.size(1), target_differences.size(1))
            self._merge_perturbation_plots([perturbation_plots[0], perturbation_plots[2]], mirna_differences.size(1), target_differences.size(1), suffix="TN-c5p", ci=True)
        else:
            # log single position perturbation of true positive samples
            tp = (y == 1) & (pred_labels == 1)
            _, _, _ = self._single_position_perturbation_ci(
                mirna_differences[tp], target_differences[tp], "TP")

            # log single position perturbation of true negative samples
            tn = (y == 0) & (pred_labels == 0)
            _, _, _ = self._single_position_perturbation_ci(
                mirna_differences[tn], target_differences[tn], "TF")

    def _merge_perturbation_plots(self, perturbation_plots, mirna_len, target_len, suffix="TP", ci=False):
        spacer = 1
        x = np.arange(mirna_len + spacer + target_len)
        fig = plt.figure(figsize=(len(x) // 4, 4))
        color = ["#4477aa", "#ff5f00", "#228833", "#aa3377"]
        plt.plot(x, np.zeros_like(x), color='black', linestyle='--')
        for i, (category, y, y_up, y_low) in enumerate(perturbation_plots):
            plt.plot(x, y, label=category, color=color[i], linewidth=2)
            if ci:
                plt.fill_between(x, y_up, y_low, color=color[i], alpha=0.3, label=f'{category} 95% CI')
        plt.xticks(x, [f"M{i+1}" for i in range(mirna_len)] + [""] * spacer + [f"T{i+1}" for i in range(target_len)], rotation=45)
        plt.xlabel("Position")
        plt.ylabel("Predicted probability difference of single position perturbation")
        plt.title(f"Single position perturbation of {suffix}")
        plt.legend()
        path = os.path.join(self.trainer.default_root_dir, "figures", f"perturbation_{suffix}.png")
        fig.savefig(path, bbox_inches='tight')
        if hasattr(self.logger.experiment, "log_image"):
            self.logger.experiment.log_image(path)
        else:
            # Optionally log locally or simply skip.
            pass
        plt.close(fig)

    def _single_position_perturbation_ci(self, mirna_differences, target_differences, suffix="TP"):
        # calculate mean and 95% CI for miRNA
        means_mirna = mirna_differences.mean(dim=0).numpy()
        stderr_mirna = stats.sem(mirna_differences.numpy(), axis=0)
        ci95_mirna = stderr_mirna * stats.t.ppf((1 + 0.95) / 2., mirna_differences.size(0) - 1)

        # calculate mean and 95% CI for target
        means_target = target_differences.mean(dim=0).numpy()
        stderr_target = stats.sem(target_differences.numpy(), axis=0)
        ci95_target = stderr_target * stats.t.ppf((1 + 0.95) / 2., target_differences.size(0) - 1)

        # plot mirna, space and target
        spacer = 1
        x = np.arange(mirna_differences.size(1) + spacer + target_differences.size(1))
        fig = plt.figure(figsize=(len(x), 15))
        y = np.concatenate([means_mirna, [np.nan] * spacer, means_target])
        y_uper = np.concatenate([means_mirna + ci95_mirna, [np.nan] * spacer, means_target + ci95_target])
        y_lower = np.concatenate([means_mirna - ci95_mirna, [np.nan] * spacer, means_target - ci95_target])
        plt.plot(x, y, '-o', label='Mean')
        plt.plot(x, np.zeros_like(x), color='black', linestyle='--')
        plt.fill_between(x, y_uper, y_lower, color='gray', alpha=0.3, label='95% CI')
        plt.xticks(x, [f'M{i + 1}' for i in range(mirna_differences.size(1))] + [""] * spacer + [f'T{i + 1}' for i in range(target_differences.size(1))], fontsize=30, rotation=45)
        plt.yticks(fontsize=40)
        path = os.path.join(self.trainer.default_root_dir, "figures", f"permutation-{suffix}.png")
        fig.savefig(path, bbox_inches='tight')
        if hasattr(self.logger.experiment, "log_image"):
            self.logger.experiment.log_image(path)
        else:
            # Optionally log locally or simply skip.
            pass
        plt.close(fig)

        return y, y_uper, y_lower

    def predict_step(self, batch, batch_idx):
        mirna_input, target_input, mirna_mask, target_mask = batch

        # get the predictions
        preds = self.model(mirna_input=mirna_input, target_input=target_input, type_input=None)
        pred_labels = (preds >= 0.5).long().detach().cpu().numpy()

        # enable gradient tracking for captum
        self.train()
        torch.set_grad_enabled(True)
        ig = IntegratedGradients(self.model)

        # TODO: not support for type_input yet
        attributions = ig.attribute(inputs=(mirna_input, target_input), target=0)
        attributions_mirna = attributions[0].permute(0, 2, 1).detach().cpu().numpy()
        attributions_target = attributions[1].permute(0, 2, 1).detach().cpu().numpy()

        self.eval()

        # get masked weights
        masked_attributions_mirna = attributions_mirna * mirna_mask.permute(0, 2, 1).cpu().numpy()
        masked_attributions_target = attributions_target * target_mask.permute(0, 2, 1).cpu().numpy()
        if len(self.model.n_encoding) == 5:
            masked_attributions_mirna = masked_attributions_mirna[:, 1:, :]
            masked_attributions_target = masked_attributions_target[:, 1:, :]

        # log path
        plot_path = os.path.join(self.trainer.default_root_dir, "figures")
        os.makedirs(plot_path, exist_ok=True)

        for i in range(batch[0].size(0)):
            # plot interpretations
            fig_mirna, _ = plot_weights(masked_attributions_mirna[i])
            path = os.path.join(self.trainer.default_root_dir, "figures", f"mirna_{batch_idx}_{i}.png")
            fig_mirna.savefig(path, bbox_inches='tight')
            self.logger.experiment.log_image(path)
            plt.close(fig_mirna)

            fig_target, _ = plot_weights(masked_attributions_target[i])
            path = os.path.join(self.trainer.default_root_dir, "figures", f"target_batch{batch_idx}_{i}.png")
            fig_target.savefig(path, bbox_inches='tight')
            self.logger.experiment.log_image(path)
            plt.close(fig_target)

        # calculate permutation importance
        multi_permutation = {}
        for window_size in range(1, min(7, batch[0].size(1)) + 1):
            # calculate probabilities for mutated mirnas
            mirna_differences = []
            for i in range(batch[0].size(1) - window_size + 1):
                # permute the i-th position of the mirna sequence with n-encoding
                mirna_perm = batch[0].clone().to(self.device)
                n_encoding_tensor = torch.tensor(self.model.n_encoding, dtype=torch.float32).to(self.device)
                n_encoding_tensor = n_encoding_tensor.unsqueeze(0).expand(window_size, -1)
                mirna_perm[:, i:i+window_size, :] = n_encoding_tensor

                # get predictions for permuted sequence
                permuted_preds = self.model(mirna_input=mirna_perm, target_input=batch[1], type_input=None)
                permuted_preds = permuted_preds.squeeze(dim=1)

                # calculate difference
                mirna_differences.append((preds - permuted_preds).detach().cpu())

            mirna_diff = torch.stack(mirna_differences, dim=1)

            # calculate probabilities for permutated target
            target_differences = []
            for i in range(batch[1].size(1) - window_size + 1):
                # permute the i-th position of the target sequence with n-encoding
                target_perm = batch[1].clone().to(self.device)
                n_encoding_tensor = torch.tensor(self.model.n_encoding, dtype=torch.float32).to(self.device)
                n_encoding_tensor = n_encoding_tensor.unsqueeze(0).expand(window_size, -1)
                target_perm[:, i:i+window_size, :] = n_encoding_tensor

                # get predictions for permuted sequence
                permuted_preds = self.model(mirna_input=batch[0], target_input=target_perm, type_input=None)
                permuted_preds = permuted_preds.squeeze(dim=1)

                # calculate difference
                target_differences.append((preds - permuted_preds).detach().cpu())

            target_diff = torch.stack(target_differences, dim=1)

            # padding
            mirna_padding = torch.full((mirna_diff.size(0), self.model.max_mirna_len - mirna_diff.size(1) + 1),
                                       float('nan'))
            target_padding = torch.full((target_diff.size(0), self.model.max_target_len - target_diff.size(1)), float('nan'))

            # concatenate
            multi_permutation[window_size] = torch.cat([mirna_diff, mirna_padding, target_diff, target_padding], dim=1).numpy()

        # plot permutation importance
        x = multi_permutation[1].shape[1]
        for i in range(batch[0].size(0)):
            fig = plt.figure(figsize=(x, 15))
            plt.axhline(y=pred_labels[i], linestyle="--")
            for k in multi_permutation:
                plt.scatter(np.arange(x), multi_permutation[k][i], label=str(k))
            plt.xticks(np.arange(1, x + 1), (
                               [f"M{i + 1}" for i in range(self.model.max_mirna_len)]
                               + [""] * 1
                               + [f"T{i + 1}" for i in range(self.model.max_target_len)]
                       ), fontsize=20)
            plt.legend()
            fig_path = os.path.join(self.trainer.default_root_dir, "figures", f"multi_permutation_{batch_idx}_{i}.png")
            plt.savefig(fig_path, bbox_inches='tight')
            self.logger.experiment.log_image(fig_path)
            plt.close()
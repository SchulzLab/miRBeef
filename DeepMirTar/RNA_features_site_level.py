import sys
sys.path.append('/usr/local/lib/python3.6/dist-packages')
import RNA
import os


##
def string_together_alignment_results(path_file):
    #extract alignment segment
    f = open(path_file)
    cons_allstr = f.readlines()
    f.close()
    loc_Q_L = []
    for i in range(len(cons_allstr)):
        if cons_allstr[i].startswith('Query_1'):
            loc_Q_L.append(i)
        elif cons_allstr[i].startswith('Lambda'):
            loc_Q_L.append(i-1)
    all_segment = []
    for j in range(len(loc_Q_L)-2):
        segment = cons_allstr[loc_Q_L[j]:loc_Q_L[j+1]-1]
        all_segment.append(segment)

    # obtain the location of seq in each line
    all_start_end = []
    for segment in all_segment:
        q_start = int(segment[0].strip().split()[1])
        q_end = int(segment[0].strip().split()[3])
        all_start_end.append([q_start, q_end])

    scope = [all_start_end[0]]
    for se in all_start_end[1:]:
        if se[0]-1 == scope[-1][1]:
            scope[-1][1] = se[1]
        else:
            scope.append(se)
    # obtain the location of seq in each line
    first_query_line = all_segment[0][0].strip()
    query_1 = first_query_line.split()[2]
    query_start = first_query_line.find(query_1)
    query_end = query_start + len(query_1)

    # string together each segment
    blast_result_dic = {}
    for k in range(21):
        blast_result_dic[str(k)] = ''
        for l in range(len(all_segment)):
            segment = all_segment[l]
            if len(segment) > k:
                line = segment[k].strip().replace(' ', '-')
                seq = line[query_start: query_end]
                blast_result_dic[str(k)] += seq
            else:
                seq = '-' * len(query_1)
                blast_result_dic[str(k)] += seq
    # delete the gap in the query, delete the corresponding base in the alignment results
    blast_result_dic_new = {}
    for k in range(21):
        blast_result_dic_new[str(k)] = ''

    for i in range(len(blast_result_dic['0'])):
        if blast_result_dic['0'][i] in ['A', 'U', 'G', 'C', 'T', 'a', 'u', 'g', 'c', 't']:
            for k in range(21):
                blast_result_dic_new[str(k)] += blast_result_dic[str(k)][i]

    blast_result_dic_new['index'] = scope
    return blast_result_dic_new


# # Methods
def seed_complementary(seq1, seq2):
    seq1 = seq1.upper().replace('T', 'U')
    seq2 = seq2.upper().replace('T', 'U')

    count_c = 0
    count_w = 0
    count_m = 0

    c = ['AU','UA','GC','CG']
    w = ['GU','UG']

    for i in range(len(seq1)):
        ss = seq1[i]+seq2[i]
        if ss in c:
            count_c += 1
        elif ss in w:
            count_w += 1
    result = {'count_c': count_c, 'count_w': count_w}
    return result


# # 1. Seed match type (26)
def seed_match_type(mir, mr_site): # 26
    mirna = mir.upper().replace('T', 'U')
    mrna = mr_site.upper().replace('T', 'U')
    c4 = ['AU', 'UA', 'GC', 'CG']
    w2 = ['GU', 'UG']
    smt_dic = {'Seed_match_8mer': 0,
               'Seed_match_8merA1': 0,
               'Seed_match_7mer1': 0,
               'Seed_match_7mer2': 0,
               'Seed_match_7merA1': 0,
               'Seed_match_6mer1': 0,
               'Seed_match_6mer2': 0,
               'Seed_match_6mer3': 0,
               'Seed_match_6mer1GU1': 0,
               'Seed_match_6mer2GU1': 0,
               'Seed_match_6mer3GU1': 0,
               'Seed_match_6mer1GU2': 0,
               'Seed_match_6mer2GU2': 0,
               'Seed_match_6mer3GU2': 0,
               'Seed_match_6mer1GU3': 0,
               'Seed_match_6mer2GU3': 0,
               'Seed_match_6mer3GU3': 0,
               'Seed_match_6mer1GU4': 0,
               'Seed_match_6mer2GU4': 0,
               'Seed_match_6mer3GU4': 0,
               'Seed_match_6mer1GU5': 0,
               'Seed_match_6mer2GU5': 0,
               'Seed_match_6mer3GU5': 0,
               'Seed_match_6mer1GU6': 0,
               'Seed_match_6mer2GU6': 0,
               'Seed_match_6mer3GU6': 0}

    # # Seed_match_8mer
    if seed_complementary(mirna[-8:], mrna[-8:])['count_c'] == 8:
        smt_dic['Seed_match_8mer'] = 1

    # # Seed_match_8merA1
    if seed_complementary(mirna[-8:-1], mrna[-8:-1])['count_c'] == 7 and mirna[-1] == 'A' and mirna[-1]+mrna[-1] not in c4:
        smt_dic['Seed_match_8merA1'] = 1
    if seed_complementary(mirna[-8:-1], mrna[-8:-1])['count_c'] == 7 and mrna[-1] == 'A'and mirna[-1]+mrna[-1] not in c4:
        smt_dic['Seed_match_8merA1'] = 1

    # # Seed_match_7mer1
    if seed_complementary(mirna[-7:], mrna[-7:])['count_c'] == 7:
        smt_dic['Seed_match_7mer1'] = 1

    # # Seed_match_7mer2
    if seed_complementary(mirna[-8:-1], mrna[-8:-1])['count_c'] == 7:
        smt_dic['Seed_match_7mer2'] = 1

    # # Seed_match_7merA1
    if seed_complementary(mirna[-7:-1], mrna[-7:-1])['count_c'] == 6 and mirna[-1] == 'A' and mirna[-1]+mrna[-1] not in c4:
        smt_dic['Seed_match_7merA1'] = 1
    if seed_complementary(mirna[-7:-1], mrna[-7:-1])['count_c'] == 6 and mrna[-1] == 'A' and mirna[-1]+mrna[-1] not in c4:
        smt_dic['Seed_match_7merA1'] = 1

    # # Seed_match_6mer1, Seed_match_6mer2, Seed_match_6mer3
    if seed_complementary(mirna[-6:], mrna[-6:])['count_c'] == 6:
        smt_dic['Seed_match_6mer1'] = 1
    if seed_complementary(mirna[-7:-1], mrna[-7:-1])['count_c'] == 6:
        smt_dic['Seed_match_6mer2'] = 1
    if seed_complementary(mirna[-8:-2], mrna[-8:-2])['count_c'] == 6:
        smt_dic['Seed_match_6mer3'] = 1

    # # Seed_match_6mer1GU1,2,3,4,5,6
    if seed_complementary(mirna[-6:], mrna[-6:])['count_c'] == 5 and seed_complementary(mirna[-6:], mrna[-6:])['count_w'] == 1:
        smt_dic['Seed_match_6mer1GU1'] = 1
    if seed_complementary(mirna[-6:], mrna[-6:])['count_c'] == 4 and seed_complementary(mirna[-6:], mrna[-6:])['count_w'] == 2:
        smt_dic['Seed_match_6mer1GU2'] = 1
    if seed_complementary(mirna[-6:], mrna[-6:])['count_c'] == 3 and seed_complementary(mirna[-6:], mrna[-6:])['count_w'] == 3:
        smt_dic['Seed_match_6mer1GU3'] = 1
    if seed_complementary(mirna[-6:], mrna[-6:])['count_c'] == 2 and seed_complementary(mirna[-6:], mrna[-6:])['count_w'] == 4:
        smt_dic['Seed_match_6mer1GU4'] = 1
    if seed_complementary(mirna[-6:], mrna[-6:])['count_c'] == 1 and seed_complementary(mirna[-6:], mrna[-6:])['count_w'] == 5:
        smt_dic['Seed_match_6mer1GU5'] = 1
    if seed_complementary(mirna[-6:], mrna[-6:])['count_c'] == 0 and seed_complementary(mirna[-6:], mrna[-6:])['count_w'] == 6:
        smt_dic['Seed_match_6mer1GU6'] = 1

    # # Seed_match_6mer2GU1,2,3,4,5,6
    if seed_complementary(mirna[-7:-1], mrna[-7:-1])['count_c'] == 5 and seed_complementary(mirna[-7:-1], mrna[-7:-1])['count_w'] == 1:
        smt_dic['Seed_match_6mer2GU1'] = 1
    if seed_complementary(mirna[-7:-1], mrna[-7:-1])['count_c'] == 4 and seed_complementary(mirna[-7:-1], mrna[-7:-1])['count_w'] == 2:
        smt_dic['Seed_match_6mer2GU2'] = 1
    if seed_complementary(mirna[-7:-1], mrna[-7:-1])['count_c'] == 3 and seed_complementary(mirna[-7:-1], mrna[-7:-1])['count_w'] == 3:
        smt_dic['Seed_match_6mer2GU3'] = 1
    if seed_complementary(mirna[-7:-1], mrna[-7:-1])['count_c'] == 2 and seed_complementary(mirna[-7:-1], mrna[-7:-1])['count_w'] == 4:
        smt_dic['Seed_match_6mer2GU4'] = 1
    if seed_complementary(mirna[-7:-1], mrna[-7:-1])['count_c'] == 1 and seed_complementary(mirna[-7:-1], mrna[-7:-1])['count_w'] == 5:
        smt_dic['Seed_match_6mer2GU5'] = 1
    if seed_complementary(mirna[-7:-1], mrna[-7:-1])['count_c'] == 0 and seed_complementary(mirna[-7:-1], mrna[-7:-1])['count_w'] == 6:
        smt_dic['Seed_match_6mer2GU6'] = 1

    # # Seed_match_6mer3GU1,2,3,4,5,6
    if seed_complementary(mirna[-8:-2], mrna[-8:-2])['count_c'] == 5 and seed_complementary(mirna[-8:-2], mrna[-8:-2])['count_w'] == 1:
        smt_dic['Seed_match_6mer3GU1'] = 1
    if seed_complementary(mirna[-8:-2], mrna[-8:-2])['count_c'] == 4 and seed_complementary(mirna[-8:-2], mrna[-8:-2])['count_w'] == 2:
        smt_dic['Seed_match_6mer3GU2'] = 1
    if seed_complementary(mirna[-8:-2], mrna[-8:-2])['count_c'] == 3 and seed_complementary(mirna[-8:-2], mrna[-8:-2])['count_w'] == 3:
        smt_dic['Seed_match_6mer3GU3'] = 1
    if seed_complementary(mirna[-8:-2], mrna[-8:-2])['count_c'] == 2 and seed_complementary(mirna[-8:-2], mrna[-8:-2])['count_w'] == 4:
        smt_dic['Seed_match_6mer3GU4'] = 1
    if seed_complementary(mirna[-8:-2], mrna[-8:-2])['count_c'] == 1 and seed_complementary(mirna[-8:-2], mrna[-8:-2])['count_w'] == 5:
        smt_dic['Seed_match_6mer3GU5'] = 1
    if seed_complementary(mirna[-8:-2], mrna[-8:-2])['count_c'] == 0 and seed_complementary(mirna[-8:-2], mrna[-8:-2])['count_w'] == 6:
        smt_dic['Seed_match_6mer3GU6'] = 1
    return smt_dic


# # 2. miRNA pairing (20+18)
def miRNA_match_position(mir, mr_site):  # 20
    mirna = mir.upper().replace('T', 'U')
    mrna = mr_site.upper().replace('T', 'U')

    AU = ['AU', 'UA']
    GC = ['GC', 'CG']
    GU = ['GU', 'UG']

    if len(mirna) < 20:
        mirna += '-' * 20
        mrna += '-' * 20
    mmp_dic = {}
    for i in range(21)[1:]:
        key = 'miRNA_match_position'+str(i+1)
        pair = mirna[-i]+mrna[-i]
        if pair in AU:
            mmp_dic[key] = 2
        elif pair in GC:
            mmp_dic[key] = 1
        elif pair in GU:
            mmp_dic[key] = 3
        elif '-' in pair:
            mmp_dic[key] = 5
        else:
            mmp_dic[key] = 4
    return mmp_dic


def miRNA_pairing_count(mir, mr_site):  # 6*3=18
    mirna = mir.upper().replace('T', 'U')
    mrna = mr_site.upper().replace('T', 'U')

    AU = ['AU', 'UA']
    GC = ['GC', 'CG']
    GU = ['GU', 'UG']
    MM = ['AA', 'AG', 'AC', 'UU', 'UC', 'GA', 'GG', 'CA', 'CU', 'CC']

    mpc_dic = {'Seed_GC': 0,
               'Seed_AU': 0,
               'Seed_GU': 0,
               'Seed_mismatch': 0,
               'Seed_bulge': 0,
               'Seed_bulge_nt': 0,
               'Total_GC': 0,
               'Total_AU': 0,
               'Total_GU': 0,
               'Total_mismatch': 0,
               'Total_bulge': 0,
               'Total_bulge_nt': 0,
               'X3p_GC': 0,
               'X3p_AU': 0,
               'X3p_GU': 0,
               'X3p_mismatch': 0,
               'X3p_bulge': 0,
               'X3p_bulge_nt': 0}
    for i in range(len(mirna)+1)[1:]:
        pair = mirna[-i] + mrna[-i]
        if pair in AU:
            mpc_dic['Total_AU'] += 1
            if -9 < i < -1:
                mpc_dic['Seed_AU'] += 1
            if i <= -9:
                mpc_dic['X3p_AU'] += 1
        elif pair in GC:
            mpc_dic['Total_GC'] += 1
            if -9 < i < -1:
                mpc_dic['Seed_GC'] += 1
            if i <= -9:
                mpc_dic['X3p_GC'] += 1
        elif pair in GU:
            mpc_dic['Total_GU'] += 1
            if -9 < i < -1:
                mpc_dic['Seed_GU'] += 1
            if i <= -9:
                mpc_dic['X3p_GU'] += 1
        elif pair in MM:
            mpc_dic['Total_mismatch'] += 1
            if -9 < i < -1:
                mpc_dic['Seed_mismatch'] += 1
            if i <= -9:
                mpc_dic['X3p_mismatch'] += 1
        elif '-' in pair:
            mpc_dic['Total_bulge_nt'] += 1
            if -9 < i < -1:
                mpc_dic['Seed_bulge_nt'] += 1
            if i <= -9:
                mpc_dic['X3p_bulge_nt'] += 1
    mirna = 'A' + mirna
    for i in range(len(mirna)+1)[1:]:
        if mirna[-i] == '-' and mirna[-i-1] != '-':
            mpc_dic['Total_bulge'] += 1
            if -9 < i < -1:
                mpc_dic['Seed_bulge'] += 1
            if i <= -9:
                mpc_dic['X3p_bulge'] += 1
    return mpc_dic


# # 3. location of target site (1)
def distance_to_end(mrna, mr_site_loc):  # 1
    dte_dic = {'Dist_to_end': round(float(len(mrna) - int(mr_site_loc[1]))/len(mrna), 4)}
    return dte_dic


# # 4. target composition (20+20+20)
def target_composition(mr_site):  # 20
    mrna = mr_site.upper().replace('T', 'U')
    count_A = 0
    count_U = 0
    count_G = 0
    count_C = 0
    count_AA = 0
    count_AU = 0
    count_AG = 0
    count_AC = 0
    count_UA = 0
    count_UU = 0
    count_UG = 0
    count_UC = 0
    count_GA = 0
    count_GU = 0
    count_GG = 0
    count_GC = 0
    count_CA = 0
    count_CU = 0
    count_CG = 0
    count_CC = 0
    for i in range(len(mrna)):
        if mrna[i] == 'A':
            count_A += 1
        elif mrna[i] == 'U':
            count_U += 1
        elif mrna[i] == 'G':
            count_G += 1
        elif mrna[i] == 'C':
            count_C += 1
    for i in range(len(mrna)-1):
        if mrna[i:i+2] == 'AA':
            count_AA += 1
        elif mrna[i:i+2] == 'AU':
            count_AU += 1
        elif mrna[i:i+2] == 'AG':
            count_AG += 1
        elif mrna[i:i+2] == 'AC':
            count_AC += 1
        elif mrna[i:i+2] == 'UA':
            count_UA += 1
        elif mrna[i:i+2] == 'UU':
            count_UU += 1
        elif mrna[i:i+2] == 'UG':
            count_UG += 1
        elif mrna[i:i+2] == 'UC':
            count_UC += 1
        elif mrna[i:i+2] == 'GA':
            count_GA += 1
        elif mrna[i:i+2] == 'GU':
            count_GU += 1
        elif mrna[i:i+2] == 'GG':
            count_GG += 1
        elif mrna[i:i+2] == 'GC':
            count_GC += 1
        elif mrna[i:i+2] == 'CA':
            count_CA += 1
        elif mrna[i:i+2] == 'CU':
            count_CU += 1
        elif mrna[i:i+2] == 'CG':
            count_CG += 1
        elif mrna[i:i+2] == 'CC':
            count_CC += 1
    all_monomer_count = count_A+count_U+count_G+count_C
    all_dimer_count = count_AA+count_AU+count_AG+count_AC+\
                      count_UA+count_UU+count_UG+count_UC+\
                      count_GA+count_GU+count_GG+count_GC+\
                      count_CA+count_CU+count_CG+count_CC
    tc_dic = {'Target_A_comp': round(float(count_A)/all_monomer_count, 4),
              'Target_U_comp': round(float(count_U)/all_monomer_count, 4),
              'Target_G_comp': round(float(count_G)/all_monomer_count, 4),
              'Target_C_comp': round(float(count_C)/all_monomer_count, 4),
              'Target_AA_comp': round(float(count_AA)/all_dimer_count, 4),
              'Target_AU_comp': round(float(count_AU)/all_dimer_count, 4),
              'Target_AG_comp': round(float(count_AG)/all_dimer_count, 4),
              'Target_AC_comp': round(float(count_AC)/all_dimer_count, 4),
              'Target_UA_comp': round(float(count_UA)/all_dimer_count, 4),
              'Target_UU_comp': round(float(count_UU)/all_dimer_count, 4),
              'Target_UG_comp': round(float(count_UG)/all_dimer_count, 4),
              'Target_UC_comp': round(float(count_UC)/all_dimer_count, 4),
              'Target_GA_comp': round(float(count_GA)/all_dimer_count, 4),
              'Target_GU_comp': round(float(count_GU)/all_dimer_count, 4),
              'Target_GG_comp': round(float(count_GG)/all_dimer_count, 4),
              'Target_GC_comp': round(float(count_GC)/all_dimer_count, 4),
              'Target_CA_comp': round(float(count_CA)/all_dimer_count, 4),
              'Target_CU_comp': round(float(count_CU)/all_dimer_count, 4),
              'Target_CG_comp': round(float(count_CG)/all_dimer_count, 4),
              'Target_CC_comp': round(float(count_CC)/all_dimer_count, 4)}
    return tc_dic


def flanking_up_composition(mrna, mr_site_loc, flank_number=70):  # 20
    mrna_full = mrna.upper().replace('T', 'U')
    mrna_up = mrna_full[max(0, mr_site_loc[0]-70):mr_site_loc[0]]
    mrna_down = mrna_full[mr_site_loc[1]+1:mr_site_loc[1]+71]
    # print len(mrna_up), len(mrna_down)

    # # Up
    count_A = 0
    count_U = 0
    count_G = 0
    count_C = 0
    count_AA = 0
    count_AU = 0
    count_AG = 0
    count_AC = 0
    count_UA = 0
    count_UU = 0
    count_UG = 0
    count_UC = 0
    count_GA = 0
    count_GU = 0
    count_GG = 0
    count_GC = 0
    count_CA = 0
    count_CU = 0
    count_CG = 0
    count_CC = 0
    for i in range(len(mrna_up)):
        if mrna_up[i] == 'A':
            count_A += 1
        elif mrna_up[i] == 'U':
            count_U += 1
        elif mrna_up[i] == 'G':
            count_G += 1
        elif mrna_up[i] == 'C':
            count_C += 1
    for i in range(len(mrna_up) - 1):
        if mrna_up[i:i + 2] == 'AA':
            count_AA += 1
        elif mrna_up[i:i + 2] == 'AU':
            count_AU += 1
        elif mrna_up[i:i + 2] == 'AG':
            count_AG += 1
        elif mrna_up[i:i + 2] == 'AC':
            count_AC += 1
        elif mrna_up[i:i + 2] == 'UA':
            count_UA += 1
        elif mrna_up[i:i + 2] == 'UU':
            count_UU += 1
        elif mrna_up[i:i + 2] == 'UG':
            count_UG += 1
        elif mrna_up[i:i + 2] == 'UC':
            count_UC += 1
        elif mrna_up[i:i + 2] == 'GA':
            count_GA += 1
        elif mrna_up[i:i + 2] == 'GU':
            count_GU += 1
        elif mrna_up[i:i + 2] == 'GG':
            count_GG += 1
        elif mrna_up[i:i + 2] == 'GC':
            count_GC += 1
        elif mrna_up[i:i + 2] == 'CA':
            count_CA += 1
        elif mrna_up[i:i + 2] == 'CU':
            count_CU += 1
        elif mrna_up[i:i + 2] == 'CG':
            count_CG += 1
        elif mrna_up[i:i + 2] == 'CC':
            count_CC += 1
    all_monomer_count = count_A + count_U + count_G + count_C
    all_dimer_count = count_AA + count_AU + count_AG + count_AC + \
                      count_UA + count_UU + count_UG + count_UC + \
                      count_GA + count_GU + count_GG + count_GC + \
                      count_CA + count_CU + count_CG + count_CC
    if all_monomer_count == 0:
        all_monomer_count +=70
    if all_dimer_count == 0:
        all_dimer_count += 70
    fuc_dic = {'Up_A_comp': round(float(count_A) / all_monomer_count, 4),
              'Up_U_comp': round(float(count_U) / all_monomer_count, 4),
              'Up_G_comp': round(float(count_G) / all_monomer_count, 4),
              'Up_C_comp': round(float(count_C) / all_monomer_count, 4),
              'Up_AA_comp': round(float(count_AA) / all_dimer_count, 4),
              'Up_AU_comp': round(float(count_AU) / all_dimer_count, 4),
              'Up_AG_comp': round(float(count_AG) / all_dimer_count, 4),
              'Up_AC_comp': round(float(count_AC) / all_dimer_count, 4),
              'Up_UA_comp': round(float(count_UA) / all_dimer_count, 4),
              'Up_UU_comp': round(float(count_UU) / all_dimer_count, 4),
              'Up_UG_comp': round(float(count_UG) / all_dimer_count, 4),
              'Up_UC_comp': round(float(count_UC) / all_dimer_count, 4),
              'Up_GA_comp': round(float(count_GA) / all_dimer_count, 4),
              'Up_GU_comp': round(float(count_GU) / all_dimer_count, 4),
              'Up_GG_comp': round(float(count_GG) / all_dimer_count, 4),
              'Up_GC_comp': round(float(count_GC) / all_dimer_count, 4),
              'Up_CA_comp': round(float(count_CA) / all_dimer_count, 4),
              'Up_CU_comp': round(float(count_CU) / all_dimer_count, 4),
              'Up_CG_comp': round(float(count_CG) / all_dimer_count, 4),
              'Up_CC_comp': round(float(count_CC) / all_dimer_count, 4)}
    return fuc_dic


def flanking_down_composition(mrna, mr_site_loc, flank_number=70):  # 20
    mrna_full = mrna.upper().replace('T', 'U')
    mrna_up = mrna_full[max(0, mr_site_loc[0] - 70):mr_site_loc[0]]
    mrna_down = mrna_full[mr_site_loc[1] + 1:mr_site_loc[1] + 71]
    # print len(mrna_up), len(mrna_down)

    # # Down
    count_A = 0
    count_U = 0
    count_G = 0
    count_C = 0
    count_AA = 0
    count_AU = 0
    count_AG = 0
    count_AC = 0
    count_UA = 0
    count_UU = 0
    count_UG = 0
    count_UC = 0
    count_GA = 0
    count_GU = 0
    count_GG = 0
    count_GC = 0
    count_CA = 0
    count_CU = 0
    count_CG = 0
    count_CC = 0
    for i in range(len(mrna_down)):
        if mrna_down[i] == 'A':
            count_A += 1
        elif mrna_down[i] == 'U':
            count_U += 1
        elif mrna_down[i] == 'G':
            count_G += 1
        elif mrna_down[i] == 'C':
            count_C += 1
    for i in range(len(mrna_down) - 1):
        if mrna_down[i:i + 2] == 'AA':
            count_AA += 1
        elif mrna_down[i:i + 2] == 'AU':
            count_AU += 1
        elif mrna_down[i:i + 2] == 'AG':
            count_AG += 1
        elif mrna_down[i:i + 2] == 'AC':
            count_AC += 1
        elif mrna_down[i:i + 2] == 'UA':
            count_UA += 1
        elif mrna_down[i:i + 2] == 'UU':
            count_UU += 1
        elif mrna_down[i:i + 2] == 'UG':
            count_UG += 1
        elif mrna_down[i:i + 2] == 'UC':
            count_UC += 1
        elif mrna_down[i:i + 2] == 'GA':
            count_GA += 1
        elif mrna_down[i:i + 2] == 'GU':
            count_GU += 1
        elif mrna_down[i:i + 2] == 'GG':
            count_GG += 1
        elif mrna_down[i:i + 2] == 'GC':
            count_GC += 1
        elif mrna_down[i:i + 2] == 'CA':
            count_CA += 1
        elif mrna_down[i:i + 2] == 'CU':
            count_CU += 1
        elif mrna_down[i:i + 2] == 'CG':
            count_CG += 1
        elif mrna_down[i:i + 2] == 'CC':
            count_CC += 1
    all_monomer_count = count_A + count_U + count_G + count_C
    all_dimer_count = count_AA + count_AU + count_AG + count_AC + \
                      count_UA + count_UU + count_UG + count_UC + \
                      count_GA + count_GU + count_GG + count_GC + \
                      count_CA + count_CU + count_CG + count_CC
    if all_monomer_count == 0:
        all_monomer_count +=70
    if all_dimer_count == 0:
        all_dimer_count += 70
    fdc_dic = {'Down_A_comp': round(float(count_A) / all_monomer_count, 4),
              'Down_U_comp': round(float(count_U) / all_monomer_count, 4),
              'Down_G_comp': round(float(count_G) / all_monomer_count, 4),
              'Down_C_comp': round(float(count_C) / all_monomer_count, 4),
              'Down_AA_comp': round(float(count_AA) / all_dimer_count, 4),
              'Down_AU_comp': round(float(count_AU) / all_dimer_count, 4),
              'Down_AG_comp': round(float(count_AG) / all_dimer_count, 4),
              'Down_AC_comp': round(float(count_AC) / all_dimer_count, 4),
              'Down_UA_comp': round(float(count_UA) / all_dimer_count, 4),
              'Down_UU_comp': round(float(count_UU) / all_dimer_count, 4),
              'Down_UG_comp': round(float(count_UG) / all_dimer_count, 4),
              'Down_UC_comp': round(float(count_UC) / all_dimer_count, 4),
              'Down_GA_comp': round(float(count_GA) / all_dimer_count, 4),
              'Down_GU_comp': round(float(count_GU) / all_dimer_count, 4),
              'Down_GG_comp': round(float(count_GG) / all_dimer_count, 4),
              'Down_GC_comp': round(float(count_GC) / all_dimer_count, 4),
              'Down_CA_comp': round(float(count_CA) / all_dimer_count, 4),
              'Down_CU_comp': round(float(count_CU) / all_dimer_count, 4),
              'Down_CG_comp': round(float(count_CG) / all_dimer_count, 4),
              'Down_CC_comp': round(float(count_CC) / all_dimer_count, 4)}
    return fdc_dic


# # 5. energy  (5)
def minimum_free_eneery(mir, mrna, mr_site_loc):  # 5
    mirna = mir.upper().replace('T', 'U').replace('-','')
    mrna_full = mrna.upper().replace('T', 'U')
    mrna_surrounding100 = mrna_full[max(0, mr_site_loc[0] - 50):mr_site_loc[1] + 50]
    mrna_site = mrna_full[mr_site_loc[0]: mr_site_loc[1]]
    mrna_seed = mrna_site[-8:-1]
    if len(mrna_site[0:-8]) > 0:
        mrna_site_3p = mrna_site[0:-8]
    else:
        mrna_site_3p = mrna_seed

    MFE_Seed = RNA.fold(mrna_seed)
    MFE_3p = RNA.fold(mrna_site_3p)
    MFE_surrounding100 = RNA.fold(mrna_surrounding100)
    duplex = RNA.duplexfold(mirna, mrna_site)
    MEF_duplex = duplex.energy

    constraint = 'x'*len(mrna_surrounding100)
    (struct, cmfe) = RNA.fold(mrna_surrounding100, constraint)

    # f = open('data3/for_feature/duplex.fa', 'w')
    # f.write('>' + 'mirna' + '\n')
    # f.write(mirna.replace('-', '') + '\n')
    # f.write('>' + 'mrna_site' + '\n')
    # f.write(mrna_site.replace('-', '') + '\n')
    # f.close()
    #
    # os.system('RNAduplex < data3/for_feature/duplex.fa > data3/for_feature/result')
    # f = open('data3/for_feature/result')
    # MEF_Duplex = float(f.readlines()[2].split()[-1].strip('(').strip(')'))
    # f.close()

    MFE = {'MEF_Seed': round(MFE_Seed[1], 4),
           'MEF_3p': round(MFE_3p[1], 4),
           'MEF_local_target': round(MFE_surrounding100[1], 4),
           'MEF_Duplex': round(MEF_duplex, 4),
           'MEF_cons_local_target': round(cmfe, 4)
           }
    return MFE


# # 6. target site accessibility  (370)
def accessibility(mrna, mr_site_loc):  # 37*10 = 370
    acc_score_matrix = list(list(i) for i in RNA.pfl_fold_up(mrna.replace('-', ''), 10, 80, 40))

    acc_score_matrix = [[0.0] * 11] * 37 + acc_score_matrix + [[0.0] * 11] * 37
    acc_score_matrix_segment = acc_score_matrix[mr_site_loc[1]+15:mr_site_loc[1]+52]
    ACC = {}
    for i in range(1, 38):
        # print acc_score_matrix_segment[i-1]
        for j in range(1, 11):
            key = 'Acc_P%s_%sth' % (str(i), str(j))
            ACC[key] = float(acc_score_matrix_segment[i - 1][j])
    return ACC


# # 7. target site conservation (160)
def conservation(mrna, mr_site_loc):
    # prepare mrna fasta file for blast
    f = open('mrna_con.fa', 'w')
    f.write(mrna.replace('-', '') + '\n')
    f.close()

    # run blast
    os.system('blastn -query mrna_con.fa '
              '-task blastn '
              '-db ./8species_database/8species_database '
              '-out conservation '
              '-outfmt 4')

    # # string together results.
    blast_result_dic_new = string_together_alignment_results('conservation')
    scope = blast_result_dic_new['index']
    # print 'scope:', scope
    # the location of which is used to represent the mrna site seq conservation

    selected_loc = [mr_site_loc[1]-30, mr_site_loc[1]+10]
    # print 'selected_loc:', selected_loc
    # print 'mr_site_loc:', mr_site_loc
    # print 'len mrna:', len(mrna)
    cons = {}
    for i in range(40):
        for j in ['A', 'U', 'G', 'C']:
            key = 'Cons_P%s_%s' % (str(i), str(j))
            cons[key] = 0.0
    count_key = 0
    for i in range(max(selected_loc[0], scope[0][0]-1), min(selected_loc[0]+40, scope[-1][-1])):
        count_A = 0
        count_U = 0
        count_G = 0
        count_C = 0
        for j in range(0, 21):
            if blast_result_dic_new[str(j)][-scope[0][0]+1+i] in ['a', 'A']:
                count_A += 1
            elif blast_result_dic_new[str(j)][-scope[0][0]+1+i] in ['U', 'u', 'T', 't']:
                count_U += 1
            elif blast_result_dic_new[str(j)][-scope[0][0]+1+i] in ['g', 'G']:
                count_G += 1
            elif blast_result_dic_new[str(j)][-scope[0][0]+1+i] in ['c', 'C']:
                count_C += 1
        key = 'Cons_P%s_' % str(count_key)
        cons[key + 'A'] = float(count_A) / (len(blast_result_dic_new)-1)
        cons[key + 'U'] = float(count_U) / (len(blast_result_dic_new)-1)
        cons[key + 'G'] = float(count_G) / (len(blast_result_dic_new)-1)
        cons[key + 'C'] = float(count_C) / (len(blast_result_dic_new)-1)
        count_key += 1
    os.remove('mrna_con.fa')
    os.remove('conservation')
    return cons


# # 8. pair hot-encoding
def pair_hot_encoding(mir, mr_site):
    mirna = mir.upper().replace('T', 'U')[-9:]
    mrna = mr_site.upper().replace('T', 'U')[-9:]

    def hot_coding(seq):
        if seq == 'A' or seq == 'a':
            he = [1, 0, 0, 0, 0]
        elif seq == 'U' or seq == 'u':
            he = [0, 1, 0, 0, 0]
        elif seq == 'T' or seq == 't':
            he = [0, 1, 0, 0, 0]
        elif seq == 'G' or seq == 'g':
            he = [0, 0, 1, 0, 0]
        elif seq == 'C' or seq == 'c':
            he = [0, 0, 0, 1, 0]
        else:
            he = [0, 0, 0, 0, 1]
        return he
    PHE = {}
    for i in range(len(mirna)):
        for j in range(5):
            key = 'MI_he_P%s_L%s' % (str(i+1), str(j+1))
            PHE[key] = hot_coding(mirna[i])[j]

    for i in range(len(mrna)):
        for j in range(5):
            key = 'MR_he_P%s_L%s' % (str(i+1), str(j+1))
            PHE[key] = hot_coding(mrna[i])[j]
    return PHE


# # 9. get all rna site level features
def get_all_RNA_site_level_features(mir, mr_site, mrna, mr_site_loc, flank_number=70):
    smt = seed_match_type(mir, mr_site)
    mmp = miRNA_match_position(mir, mr_site)
    mpc = miRNA_pairing_count(mir, mr_site)
    dte = distance_to_end(mrna, mr_site_loc)
    tc = target_composition(mr_site)
    fuc = flanking_up_composition(mrna, mr_site_loc, flank_number=flank_number)
    fdc = flanking_down_composition(mrna, mr_site_loc, flank_number=flank_number)
    mfe = minimum_free_eneery(mir, mrna, mr_site_loc)
    acc = accessibility(mrna, mr_site_loc)
    cons = conservation(mrna, mr_site_loc)
    phe = pair_hot_encoding(mir, mr_site)

    site_features = {}
    for key in smt.keys():
        site_features[key] = smt[key]

    for key in mmp.keys():
        site_features[key] = mmp[key]

    for key in mpc.keys():
        site_features[key] = mpc[key]

    for key in dte.keys():
        site_features[key] = dte[key]

    for key in tc.keys():
        site_features[key] = tc[key]

    for key in fuc.keys():
        site_features[key] = fuc[key]

    for key in fdc.keys():
        site_features[key] = fdc[key]

    for key in mfe.keys():
        site_features[key] = mfe[key]

    for key in acc.keys():
        site_features[key] = acc[key]

    for key in cons.keys():
        site_features[key] = cons[key]

    for key in phe.keys():
        site_features[key] = phe[key]

    return site_features

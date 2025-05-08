

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


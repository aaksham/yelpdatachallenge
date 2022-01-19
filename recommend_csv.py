import sys
import csv
import numpy
import pandas
import argparse
import model_utils
from sklearn.metrics.pairwise import cosine_similarity

def rank_candidates(user_vector,biz_vectors,bizids,biznames,limit=10):
    dim=user_vector.shape[0]
    candidate_vecs=numpy.zeros((len(bizids),dim))
    for i,bizv in enumerate(biz_vectors):
        candidate_vecs[i,:]=bizv
    user_vec=user_vector.reshape(1,-1)
    #print(candidate_vecs.shape)
    cos_sim_vec=cosine_similarity(user_vec,candidate_vecs)
    orig_ranks=numpy.argsort(cos_sim_vec)
    rev_orig_ranks=orig_ranks[0,:][::-1]
    result=[]
    counter=0
    for r in rev_orig_ranks:
        e=(bizids[r],biznames[r])
        result.append(e)
        counter+=1
        if counter==limit: break
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--test_csv','-t',type=str,required=True,help="Input test csv with user ids")
    args=parser.parse_args()
    test_file=args.test_csv
    test_data=pandas.read_csv(test_file)
    test_header=test_data.columns
    if 'user_id' not in test_header:
        print('"user_id" not found. See example/test*.csv for input format.')
        sys.exit(1)
    if 'city' not in test_header:
        print('"city" not found. See example/test*.csv for input format.')
        sys.exit(1)

    user_trove=pandas.read_pickle('../models/user_trove.pkl')
    biz_trove=pandas.read_pickle('models/restaurant_trove.pkl')
    rec_record=[]
    for i in range(test_data.shape[0]):
        testid=test_data.iloc[i]['user_id']
        #print(testid)
        test_city=test_data.iloc[i]['city']
        try:
            user_vector = user_trove[user_trove['user_id'] == testid]['user_vector'].iloc[0]
        except:
            print("No past record for user.")
            sample_review=test_data.iloc[i]['sample_user_review']
            if (not pandas.isna(sample_review)) and len(sample_review)>0:
                user_vector=model_utils.calculate_user_vector(sample_review)
            else:
                biz_candidates = biz_trove[biz_trove['city'] == test_city]
                biz_candidates=biz_candidates.sort_values(by=['business_stars'])
                biz_candidates=biz_candidates.head(5)
                bizids_c = biz_candidates['business_id'].to_list()
                biznames_c = biz_candidates['name'].to_list()
                recs=tuple(zip(bizids_c,biznames_c))
                rec_record.append((testid, recs))
                print("Recommendations for user " + testid)
                print(recs)
                continue
        biz_candidates = biz_trove[biz_trove['city'] == test_city]
        bizids_c = biz_candidates['business_id'].to_list()
        bizvectors_c = biz_candidates['business_vector']
        biznames_c = biz_candidates['name'].to_list()
        recs = rank_candidates(user_vector, bizvectors_c, bizids_c, biznames_c)
        print("Recommendations for user "+testid)
        print(recs)
        rec_record.append((testid,recs))

    f=open('output.csv','w')
    writer=csv.writer(f)
    writer.writerow(["user_id","recommendations"])
    for el in rec_record:
        writer.writerow([el[0],el[1]])
    f.close()

if __name__=='__main__':
    main()
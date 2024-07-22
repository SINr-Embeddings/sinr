import sinr.graph_embeddings

def get_stereotypes_for_diffvector(word,model1_commonref,model2_commonref, topk_dims, n_stereotypes):
    v1=model1_commonref.get_my_vector(word)
    v2=model2_commonref.get_my_vector(word)
    diff = v2-v1
    diff_vect = dict()
    for idx,i in enumerate(diff) :
        if i !=0 :
            diff_vect[idx]=i
    srtd=sorted(diff_vect, key=lambda dict_key: abs(diff_vect[dict_key]))
    srtd.reverse()
    pouet=list(srtd)
    out = [("value :"+str(diff_vect[idx]),model1_commonref.get_dimension_stereotypes_idx(idx, n_stereotypes).with_value().get_dict()) for idx in pouet[:topk_dims]]
    return out

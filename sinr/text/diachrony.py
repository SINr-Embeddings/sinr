import sinr.graph_embeddings

def get_stereotypes_variation(model1, model2, k:int=10):
    """Return a list of variation by dimensions according to the varnn of Pierrejean adapted to stereotypes.
    Both models should be aligned !
    It is 1 - (the intersection of the k stereotypes of k stereotypes for both models)/k
    The more it is close to 0, the more the two models are similar according to their stereotypes
    :type model1:SINrVectors
    :type model2:SINrVectors
    :type k:int
    :rtype: list of tuple (idx dim, variation dim)
    
    """
    l = list()
    for idx in tqdm(range(model1.get_number_of_dimensions()), desc = 'Computing variations of stereotypes for dimensions', leave = False):
        stereotypes1 = {v for (u,v) in model1.get_dimension_stereotypes_idx(idx, topk=k).get_dict()["stereotypes"]}
        stereotypes2 = {v for (u,v) in model2.get_dimension_stereotypes_idx(idx, topk=k).get_dict()["stereotypes"]}
        var_dim = 1 - (len(stereotypes1.intersection(stereotypes2)) / k)
        l.append((idx, var_dim))
    return l

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

def FindPartitions(RndRFModel, TIn, BlockVar, NNFacts):
    """ Find partitions associated to each sample in the original dataset. As some samples are missing in a tree training
    set, the final leaf does not necessarily contain the sample target
    Input:
    RndRFModel: Random Forest model (sklearn)
    TI: features training set  <array(NSamples, Nfeatures)>
    TOut: targets training set <array(Nfeatures)>
    Output:
    Bounds: narray <tree, sample, feature, bounds(0=lower, 1=upper)>
    BlockVar: Dictionary containing the design variable informations
    @author: leonesio
    """
    import numpy as np  # Library for numerical functions
    print('Partitions computation')
    N_Dts = len(RndRFModel)  # Number of trees in the Random Forest
    N_Samples = TIn.shape[0]  # Number of samples used to train the Random Forest, that are also the number of leaves
    N_features = TIn.shape[1]  # Dimension of the input vector / number of features
    KeysList = list(BlockVar.keys())  # List with the name of the input variables (it may be 'none' if there is no
    Bounds = np.zeros((N_Dts, N_Samples, N_features, 2))  # Initalization of the bounds tensor
    if BlockVar != None:  # Initialization default bounds from the BlockVar structure
        for feat in range(len(KeysList)):
            Bounds[:, :, feat, 0] = np.ones((N_Dts, N_Samples)) * BlockVar[KeysList[feat]][
                'lb']  # Element 0 stays for lower bound
            Bounds[:, :, feat, 1] = np.ones((N_Dts, N_Samples)) * BlockVar[KeysList[feat]][
                'ub']  # Element 1 stays for upper bound
    else:  # Default choice (-Inf, +Inf)
        Bounds[:, :, :, 0] = -np.ones((N_Dts, N_Samples, N_features)) * np.inf
        Bounds[:, :, :, 1] = np.ones((N_Dts, N_Samples, N_features)) * np.inf

    # For Dependent features
    for feat in range(len(KeysList),
                      N_features):  # This is for the feature that are dependent, according with the Physic-enhanced framework: the bounds are given by the input data
        Bounds[:, :, feat, 0] = np.ones((N_Dts, N_Samples)) * min(TIn[:, feat])
        Bounds[:, :, feat, 1] = np.ones((N_Dts, N_Samples)) * max(TIn[:, feat])
    for k in range(len(RndRFModel)):  # range(len(RndRFModel)): # Get the decision Trees
        for w in range(TIn.shape[0]):  # Get the samples of the training set
            leaf_id = RndRFModel[k].apply(TIn[w, :].reshape(1, -1))  # Get the id of the leaf
            node_indicator = RndRFModel[k].decision_path(
                TIn[w, :].reshape(1, -1))  # Info about the path: list of nodes to the leaf
            node_index = node_indicator.indices[
                         node_indicator.indptr[0]: node_indicator.indptr[1]]  # Get the array of the nodes id
            for node_id in node_index:  # Scan the path nodes
                if leaf_id == node_id:  # Stop the search at the leaf
                    continue
                feat = RndRFModel[k].tree_.feature[
                    node_id]  # Get the index of the feature producing the split in the current node
                if TIn[w, feat] <= RndRFModel[k].tree_.threshold[
                    node_id]:  # (for convention the threshold is the second member of a "<" inequality, so it introdure an upper bound see https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
                    if feat < 12:
                        Bounds[k, w, feat, 1] = RndRFModel[k].tree_.threshold[node_id] * NNFacts['InStd'][feat] + \
                                                NNFacts['InMean'][
                                                    feat]  # The inequality is verified, the upper bounds is assigned (normalized features)
                    else:
                        Bounds[k, w, feat, 1] = RndRFModel[k].tree_.threshold[
                            node_id]  # The inequality is verified, the upper bound is assigned (not normalized features)
                else:
                    if feat < 12:
                        Bounds[k, w, feat, 0] = RndRFModel[k].tree_.threshold[node_id] * NNFacts['InStd'][feat] + \
                                                NNFacts['InMean'][
                                                    feat]  # The inequality is not verified, the lower bound is assigned (normalized features)
                    else:
                        Bounds[k, w, feat, 0] = RndRFModel[k].tree_.threshold[
                            node_id]  # The inequality is not verified, the lower bound is assigned (not normalized features)
    return Bounds

def FindPartitionsComp(Bounds, Constr):
    """ Find partitions that are compatible with the constraints
    Input:
    Bounds: Bounds tensor [Trees, Samples, feat, 0/1]
    Constr: Constraints definition (dict('feat':<array of indices>, 'vals':<arrays of values>))
    Output:
    Compat: Incident matrix with compatible partitions (1 if the partition is compatible, 0 otherwise)

    @author: leonesio
    """
    import numpy as np  # Library for numerical functions
    Compat=np.empty(Bounds[:,:,0,0].shape)
    print('Compatible Partitions computation')
    for k in range(Compat.shape[0]): # Iteration on the trees
        for s in range(Compat.shape[1]):
            FlagComp = 1
            for f in range(len(Constr['feats'])):
                if ((Bounds[k,s,Constr['feats'][f],0]>Constr['vals'][f]) | (Bounds[k,s,Constr['feats'][f],1]<Constr['vals'][f])):
                    FlagComp=0
                    break
            Compat[k,s]=FlagComp
    return Compat

def FindCompSetIneq(Bounds,lb,ub):
    """ Find partitions that are compatible with the constraints.
    The constraints are expressed in the form: lb <= x <= ub

    Input:
    Bounds: Bounds tensor [Trees, Samples, feat, 0/1] (0-> lower bound, 1-> upper bound)
    lb, ub: vectors of constraints for each feature. No constraints are dealt using numpy.inf and numpy.inf
    Output:
    Compat: Incident matrix with compatible partitions (1 if the partition is compatible, 0 otherwise)

    @author: leonesio
    """
    import numpy as np  # Library for numerical functions
    Compat=np.empty(Bounds[:,:,0,0].shape)
    print('Compatible Partition sets computation (parallel constraints)')
    for k in range(Compat.shape[0]): # Iteration on the trees
        Compat[k,:]=np.all((lb<=Bounds[k,:,:,1]) & (ub>=Bounds[k,:,:,0]),axis=1).astype(int)
    return Compat

def FindCompSetIneqV2(Bounds,lb,ub,*args):
    """ Find partitions that are compatible with the constraints.
    The constraints are expressed in the form: lb <= x <= ub
    and in lbA <= Ax <= ubA
    Input:
    Bounds: Bounds tensor [Trees, Samples, feat, 0/1] (0-> lower bound, 1-> upper bound)
    lb, ub: vectors of constraints for each feature. No constraints are dealt using numpy.inf and numpy.inf
    args (optional): args[0]: A matrix for linear constraints,
                     args[1]: lbA vector
                     args[2]: lbA vector
    Output:
    Compat: Incident matrix with compatible partitions (1 if the partition is compatible, 0 otherwise)

    @author: leonesio
    """
    import numpy as np  # Library for numerical functions
    A1=np.identity(len(lb))
    if not args:
        A = A1
        lbTot = lb
        ubTot = ub
    else:
        A = np.concatenate((A1, args[0]))
        lbTot = np.concatenate((lb, args[1]))
        ubTot = np.concatenate((ub, args[2]))
    Compat=np.empty(Bounds[:,:,0,0].shape)
    print('Compatible partition sets (parallel constraints)')
    for k in range(Compat.shape[0]): # Iteration on the trees
        Compat[k,:]=np.all((lbTot<=np.matmul(A,Bounds[k,:,:,1].transpose()).transpose()) & (ubTot>=np.matmul(A,Bounds[k,:,:,0].transpose()).transpose()),axis=1).astype(int)
    return Compat

def FindCompSetIneqV3(Bounds,lb,ub,*args):
    """ Find partitions that are compatible with the constraints.
    The constraints are expressed in the form: lb <= x <= ub
    and (optionally) in lbA <= Cx <= ubA
    Input:
    Bounds: Bounds tensor [Trees, Samples, feat, 0/1] (0-> lower bound, 1-> upper bound)
    lb, ub: vectors of constraints for each feature. No constraints are dealt using numpy.inf and numpy.inf
    args (optional, but, if used, all 3 must provided): args[0]: C matrix for linear constraints,
                     args[1]: lbA vector
                     args[2]: ubA vector
    Output:
    Compat: Incident matrix with compatible partitions (1 if the partition is compatible, 0 otherwise)

    @author: leonesio
    """
    import numpy as np  # Library for numerical functions
    from scipy.optimize import linprog as linprog
    if not args:
        Compat=FindCompSetIneq(Bounds,lb,ub) # Compute compatible sets considering parallel constraints only
    else:
        COrig = np.array(args[0]) # Reassigning otpional inputs
        lbA = np.array(args[1]) # ''
        ubA = np.array(args[2]) # ''
        n_features=Bounds.shape[2] # Features number
        A = np.concatenate((-np.identity(n_features),np.identity(n_features)))
        C = np.concatenate((-COrig, COrig)) # Redefining constraints matrix for one-sided inequality
        Aub = np.concatenate((A,C))  # Overall constraint matrix
        d = np.concatenate((-lbA, ubA)) #  Redefining linear constraints vector for one-sided inequality
        c=np.ones(n_features) # Dummy coefficients for LP
        Compat = FindCompSetIneq(Bounds, lb, ub) # Compute compatible sets considering parallel constraints only
        nonzero_indices= np.nonzero(Compat) # Find the nonzeros indeces
        print('Compatible partition sets (linear constraints)')
        for row, col in zip(nonzero_indices[0], nonzero_indices[1]): # Iterate on the compatible sets (w.r.t. parallel constraints)
            bj=np.concatenate((-Bounds[row,col,:,0], Bounds[row,col,:,1])) # Find j-th set limits
            bub=np.concatenate((bj,d)) # Compose the overall LP constraints vector
            Sol = linprog(c,A_ub=Aub,b_ub=bub,bounds=(-np.inf,np.inf)) # LP solution
            if Sol.status==2: # If problem is unfeasible (state=2) mark incompatible set
                Compat[row,col]=0
    return Compat

def chebyshev_ball(A, b):
    """
    Chebyshev ball finds the largest ball inside of a polytope defined by Ax <= b
    This is solved by the following LP

    min{x,r} -r

    st:
            Ax + ||A_i||r <= b
            r >=0

    :param A: LHS Constraint Matrix
    :param b: RHS Constraint column vector
    :return: The radius is the last number in the solution of the solver object if it is feasible, and the center.

    """
    import numpy as np  # Library for numerical functions
    from scipy.optimize import linprog as linprog

    b=b.reshape(-1,1)

    c = np.zeros((A.shape[1] + 1, 1))
    c[A.shape[1]][0] = -1
    normA=np.linalg.norm(A,axis=1)

    A_ball = np.block([[A, normA.reshape(-1,1)], [c.T]])
    b_ball = np.concatenate((b, np.zeros((1, 1))))

    Sol = linprog(c, A_ub=A_ball, b_ub=b_ball, bounds=(-np.inf, np.inf))  # LP solution
    print(Sol)

    return Sol.x[:A.shape[1]],Sol.x[-1]

def PointPolytopeProjection(A, b, point):
    """
    Chebyshev ball finds the largest ball inside of a polytope defined by Ax <= b
    This is solved by the following LP

    min{x,r} -r

    st:
            Ax + ||A_i||r <= b
            r >=0

    :param A: LHS Constraint Matrix
    :param b: RHS Constraint column vector
    :return: The radius is the last number in the solution of the solver object if it is feasible, and the center.

    """
    import numpy as np  # Library for numerical functions
    from scipy.optimize import minimize
    from scipy.optimize import LinearConstraint

    def objectiveFun(x,*args):
        Dist=np.linalg.norm(x-args[0])
        return Dist

    linear_constraints=LinearConstraint(A,lb=-np.inf*np.ones(A.shape[0]), ub=b)
    initial_guess=point

    Sol = minimize(objectiveFun,initial_guess, args=(point), constraints=linear_constraints, method='trust-constr')  # LP solution

    return Sol.x,Sol.fun



'''def FindPartitionsV2(RndRFModel, TIn, BlockVar,NNFacts):
    """ Find partitions associated to each sample in the original dataset. As some samples are missing in a tree training
    set, the final leaf does not necessarily contain the sample target
    Input:
    RndRFModel: Random Forest model (sklearn)
    TI: features training set  <array(NSamples, Nfeatures)>
    TOut: targets training set <array(Nfeatures)>
    Output:
    Bounds: narray <tree, sample, feature, bounds(0=lower, 1=upper)>
    BlockVar: Dictionary containing the design variable informations
    @author: leonesio
    """
    import numpy as np
    print('Partitions computation')
    N_Dts = len(RndRFModel)
    N_Samples = TIn.shape[0]
    N_features = TIn.shape[1]
    KeysList=list(BlockVar.keys())
    Bounds = np.zeros((N_Dts, N_Samples, N_features, 2))
    if BlockVar!=None:
        for feat in range(len(KeysList)):
            Bounds[:, :, feat, 0]=np.ones((N_Dts, N_Samples)) * BlockVar[KeysList[feat]]['lb']
            Bounds[:, :, feat, 1] = np.ones((N_Dts, N_Samples)) * BlockVar[KeysList[feat]]['ub']
    else:
        Bounds[:, :, :, 0] = -np.ones((N_Dts, N_Samples, N_features)) * np.inf
        Bounds[:, :, :, 1] = np.ones((N_Dts, N_Samples, N_features)) * np.inf

    # For Dependent features
    for feat in range(len(KeysList),N_features):
        Bounds[:, :, feat, 0] = np.ones((N_Dts, N_Samples)) * min(TIn[:,feat])
        Bounds[:, :, feat, 1] = np.ones((N_Dts, N_Samples)) * max(TIn[:,feat])
    Q=[]
    for k in range(len(RndRFModel)): #range(len(RndRFModel)): # Get the decision Trees
        for w in range(TIn.shape[0]):
                leaf_id=RndRFModel[k].apply(TIn[w,:].reshape(1,-1))
                node_indicator=RndRFModel[k].decision_path(TIn[w,:].reshape(1,-1)) # Info about the path
                node_index = node_indicator.indices[node_indicator.indptr[0] : node_indicator.indptr[1]]
                l=0 # It tracks the level of the node in the tree
                Nl=[]
                VectInf = np.empty()
                for node_id in node_index
                    if leaf_id==node_id: #Stop the search at the leaf
                        continue
                    feat=RndRFModel[k].tree_.feature[node_id]

                    if TIn[w,feat] <= RndRFModel[k].tree_.threshold[node_id]:
                        if feat < 12:
                            Bounds[k, w, feat, 1] = RndRFModel[k].tree_.threshold[node_id] * NNFacts['InStd'][feat] + \
                                                    NNFacts['InMean'][feat]
                        else:
                            Bounds[k, w, feat, 1] = RndRFModel[k].tree_.threshold[node_id]
                    else:
                        if feat < 12:
                            Bounds[k, w, feat, 0] = RndRFModel[k].tree_.threshold[node_id] * NNFacts['InStd'][feat] + \
                                                    NNFacts['InMean'][feat]
                        else:
                            Bounds[k, w, feat, 0] = RndRFModel[k].tree_.threshold[node_id]
                    l+=1
                s+=1
                Q[k]

    return Bounds'''


def FixMILPPar(rf):
    """

    :param rf: Random Forest to be optimized
    :return: [a_i^t] matrix and {b_i^t} vector of the reference article [Biggs2018]
    """
    import numpy as np

    # Initializations
    a = []  # List of matrices (matrices row dimension can vary)
    b = []  # List of vectors (vector length can vary)
    s = []  # List of leaves values

    for k in range(len(rf)):  # scan trees
        a.append(np.zeros((rf[k].tree_.node_count, rf[k].n_features_in_)))  # For each tree, initialize an incident
        # matrix pickin the right feature for the node split
        a[k][range(rf[k].tree_.node_count), rf[k].tree_.feature] = 1  # Fill the incident matrix
        b.append(rf[k].tree_.threshold)  # Append the thresholds vector

        # Start the part identifying if a node is a leaf
        node_depth = np.zeros(shape=rf[k].tree_.node_count, dtype=np.int64)
        is_leaves = np.zeros(shape=rf[k].tree_.node_count, dtype=bool)
        stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
        st = np.empty((0, 2))
        while len(stack) > 0:
            # `pop` ensures each node is only visited once
            node_id, depth = stack.pop()
            node_depth[node_id] = depth

            # If the left and right child of a node is not the same we have a split
            # node
            is_split_node = rf[k].tree_.children_left[node_id] != rf[k].tree_.children_right[node_id]
            # If a split node, append left and right children and depth to `stack`
            # so we can loop through them
            if is_split_node:
                stack.append((rf[k].tree_.children_left[node_id],
                              depth + 1))  # Depth associated to the node, just for debug (not used)
                stack.append((rf[k].tree_.children_right[node_id],
                              depth + 1))  # Depth associated to the node, just for debug (not used)
            else:
                is_leaves[node_id] = True
                st = np.append(st, [[node_id, rf[k].tree_.value[node_id, 0, 0]]], axis=0)
        s.append(st)
    return a, b, s


def OptimRFMILP(RF, lb_x, ub_x):
    """
    :param RF: Random Forest to be optimized
    :param lb_x: lower bounds on design var
    :param ub_x: Constraints matrix
    :return: x, F (argument vector e function value)
    """

    # Referring to the work of Biggs et al. (2018) we have to obtain the quantities [a] {b} and S (see the paper) from
    # the random forest. This is done by FixMILPPar from this same library.
    from RFOptimLib import FixMILPPar
    from scipy.optimize import milp as milp  # Function to perform the optimization
    from scipy.optimize import Bounds as Bounds
    from scipy.optimize import LinearConstraint as LinearConstraint
    from scipy.sparse import csr_matrix
    import numpy as np

    # Sub-function to find epsilon
    def find_eps(thrs):
        th1=np.matmul(np.ones((len(thrs), 1)), np.transpose(thrs).reshape(1, -1))
        th2 = np.transpose(th1)
        thdiff=abs(th1-th2)
        thdiff[thdiff==0]=np.inf
        epsilon=thdiff.min()/2
        return epsilon
    a, b, S = FixMILPPar(RF)  # Compute the parameters [a] {b} and S from the RF
    # Initialization
    tot_n_count = 0  # Initializing the number of total internal nodes of the whole forest
    int_nodes_N = []
    nodes_L = []  # Initializing the list of the left nodes index for each tree
    nodes_R = []  # Initializing the list of the right nodes index for each tree
    int_nodes = []  # Initializing the list of the internal nodes index for each tree
    int_nodes_LR=[] # Initializing the list of the left ans right internal nodes index for each tree
    epsilon=[] # Initialization of the quantity to deal with inequality
    n_features = RF[0].tree_.n_features  # Number of features
    n_trees=len(RF) # number of trees
    aMat = np.empty((0, n_features), dtype=np.int8)  # Initialization of a matrix Internal nodes are not known a priori
    bVect = np.empty((0))  # Initialization off b vector Internal nodes are not known a priori
    for t in range(n_trees):
        nodes_L.append(RF[t].tree_.children_left)  # list of the left internal nodes index for each tree
        nodes_R.append(RF[t].tree_.children_right)  # list of the right internal nodes index for each tree
        int_nodes.append(np.where(nodes_L[t] != -1)[0])  # internal nodes
        int_nodes_LR.append(np.append(nodes_L[t][nodes_L[t] != -1], nodes_R[t][nodes_R[t] != -1]))
        int_nodes_N.append(len(int_nodes[t]))  # number of internal nodes for the t-th tree (excluding root)
        tot_n_count += 2*int_nodes_N[t]  # Tot_count is given by the cardinality of the set of left and right internal nodes
        aMat = np.append(aMat, a[t][int_nodes[t], :].astype(np.int8),
                         axis=0)  # Global incident matrix for feature selection
        bVect = np.append(bVect, b[t][int_nodes[t]], axis=0)  # Global threshold vector

        epsilon.append(find_eps(b[t][int_nodes[t]]))  # epsilon to transform >= condition in > condition (to be determined clevererly)
        print("epsilon value: "+str(t)+" --> "+str(epsilon[t]))

    M = int(np.ceil(np.amax(np.matmul(aMat, np.amax([abs(lb_x), abs(ub_x)] ,axis=0)) + bVect)))  # Definition of coefficient for the big M-method (in the theory paper)
    print("M value: "+str(M))

    cV = np.zeros((tot_n_count + n_features))  # Initialize c matrix (definining the linear relationship to be minimized)
    lb_constr1 = np.empty((0)) # Initializing polytopic constraints vector
    ub_constr1 = np.empty((0)) # Initializing polytopic constraints vector
    lb_constr2 = np.zeros((int(tot_n_count/2)))  # Initializing polytopic constraints vector
    ub_constr2 = np.zeros((int(tot_n_count/2)))  # Initializing polytopic constraints vector
    lb_constr3 = np.ones((n_trees))  # Initializing polytopic constraints vector
    ub_constr3 = np.ones((n_trees))  # Initializing polytopic constraints vector
    A_mat1 = np.zeros((tot_n_count, tot_n_count + n_features))
    A_mat2 = np.zeros((int(tot_n_count/2), tot_n_count + n_features))
    A_mat3 = np.zeros((n_trees, tot_n_count + n_features))
    # Structure of the unknowns column vector [x (dim n_features),q (all nodes)]
    # q-->[<T1 all nodes increasing index>, <T2 all nodes increasing index> ... ]
    TotCt = n_features
    TotCt_half = n_features
    for t in range(n_trees):  # Put the values of S in the right position (for each internal nodes, I have right and left child)
        ParLeafL=np.intersect1d(nodes_L[t],S[t][:, 0],return_indices=True)# Find the indices of the parents of the left leaf nodes
        ParLeafR=np.intersect1d(nodes_R[t],S[t][:, 0],return_indices=True) # Find the indices of the parents of the left leaf nodes
        IdL = np.intersect1d(int_nodes[t],  ParLeafL[1], return_indices=True)  # Identification of the parent whose the leaf is the left child [0]: values, [1]: indices
        IdR = np.intersect1d(int_nodes[t], ParLeafR[1], return_indices=True)  # Identification of the parent whose the leaf is the right child [0]: values, [1]: indices
        cV[TotCt + IdL[1][IdL[2]]] = S[t][ParLeafL[2], 1]  # Put scores in the left position when the leaf in "left" (for each tree, the sequence is left then right) (other elements=0)
        cV[TotCt + int_nodes_N[t] + IdR[1][IdR[2]]] = S[t][ParLeafR[2], 1]  # Put scores in the right position when the leaf in "right" for each tree, the sequence is left then right)
        lb_constr1 = np.append(lb_constr1, np.append(np.ones((int_nodes_N[t])) * -np.inf, -M + b[t][int_nodes[t]]+epsilon[t]))  # lower bounds in matrix constraints A1
        ub_constr1 = np.append(ub_constr1, np.append(M + b[t][int_nodes[t]], np.ones((int_nodes_N[t])) * np.inf))  # upper bounds in matrix constraints A1
        A_mat1[TotCt - n_features:TotCt - n_features + 2 * int_nodes_N[t], 0:n_features] = np.append(
            a[t][int_nodes[t], :],
            a[t][int_nodes[t], :], axis=0)  # insert [a] matrix
        A_mat1[TotCt - n_features:TotCt - n_features + int_nodes_N[t], TotCt:TotCt + 2*int_nodes_N[t]] = np.append(
            M * np.identity((int_nodes_N[t])), np.zeros((int_nodes_N[t], int_nodes_N[t])), axis=1)
        A_mat1[TotCt - n_features + int_nodes_N[t]:TotCt - n_features + 2*int_nodes_N[t],
        TotCt:TotCt + 2 * int_nodes_N[t]] = np.append(np.zeros((int_nodes_N[t], int_nodes_N[t])),
                                                  -M * np.identity((int_nodes_N[t])), axis=1)
        # Definition of matrix A2, (sum of children must be equal to the parent)
        for tt in range(int_nodes_N[t]):
            if tt==0:
                A_mat2[TotCt_half + tt - n_features, TotCt] = 1
                A_mat2[TotCt_half + tt - n_features, TotCt+int_nodes_N[t]] = 1
                lb_constr2[TotCt_half + tt - n_features]=1
                ub_constr2[TotCt_half + tt - n_features]=1
            else:
                A_mat2[TotCt_half + tt - n_features, TotCt + np.where(int_nodes_LR[t]==int_nodes[t][tt])[0]] = 1
                A_mat2[TotCt_half + tt - n_features, TotCt + np.where(int_nodes_LR[t]==nodes_L[t][int_nodes[t][tt]])[0]] = -1
                A_mat2[TotCt_half + tt - n_features, TotCt + np.where(int_nodes_LR[t]==nodes_R[t][int_nodes[t][tt]])[0]] = -1
        # Matrix for the leaf constraints (Only one leaf is active)
        A_mat3[t, TotCt + IdL[1]] = 1 # (only one leaf can be active)
        A_mat3[t, TotCt + int_nodes_N[t] + IdR[1]] = 1 # (only one leaf can be active)
        TotCt += 2 * int_nodes_N[t]  # increment the counter with left and right indices
        TotCt_half+=int_nodes_N[t] #  # increment the counter with nodes indices
        print('Parsed tree n.'+str(t))
    cV=cV/n_trees
    lb_constr = np.concatenate((lb_constr1,lb_constr2,lb_constr3))  # Equality constraints to 1 (lb and ub both equal to 1)
    ub_constr = np.concatenate((ub_constr1,ub_constr2,ub_constr3))  # Equality constraints to 1 (lb and ub both equal to 1)
    integrality = np.append(np.zeros((n_features)),
                            np.ones((tot_n_count)))  # The X are float (0) and q are integers (1)
    lb = np.append(lb_x, np.zeros(
        (tot_n_count)))  # appending the lower bounds on design variable to those of aux variables [0]
    ub = np.append(ub_x, np.ones(
        (tot_n_count)))  # appending the upper bounds on design variable to those of aux variables [1]
    bounds = Bounds(lb, ub,
                    keep_feasible=False)  # Bounds must be defined both for the design variables (lb_x and ub_x) and for the q
    # Feasibility is guaranteed only at the end
    A_matr = np.concatenate((A_mat1, A_mat2, A_mat3))
    A_matr_spr = csr_matrix(A_matr)  # Define A_matr as sparse to save memory

    constraints = LinearConstraint(A_matr_spr, lb=lb_constr, ub=ub_constr,
                                   keep_feasible=False)  # Must be represented by the matrix A appearing in the NOTES.
    # Then, transalted into a scipy object
    res = milp(c=cV, integrality=integrality, bounds=bounds, constraints=constraints,
               options={"presolve": True, "disp": True})
    return res

def InterSectBounds(Bounds, IncMat):
    """
     Intersect the bounds contained in Bounds structure (see below for the definition) according to the incidence matrix. Each sample (with the
     associated bounds must be associated to at leat one tree, otherwise no intersection is possible!
     :param Bounds: Bounds structure: [N_trees, Samples, features, bounds<0:lb,1:ub>] or [N_trees][Samples, features, bounds<0:lb,1:ub>]
     :param IncMat: Incidence matrix giving the correspondence between the trees and samples used to train the tree (matrix or
                                                        list of vectors). If "None", the intersection is made for all the samples.
     :return NewBounds: structure [Samples, features, bounds<0:lb,1:ub>]
    """
    import numpy as np

    if type(Bounds) is np.ndarray:
        if Bounds.ndim==4:
            NewBounds = np.empty((Bounds.shape[1:]))
            if IncMat is None:
                NewBounds[:, :, 0] = np.max(Bounds[:, :, :, 0], axis=0)
                NewBounds[:, :, 1] = np.min(Bounds[:, :, :, 1], axis=0)
            else:
                for k in range(IncMat.shape[0]):
                    NewBounds[k, :, 0] = np.max(Bounds[np.logical_not(IncMat[k, :].astype(bool)), k, :, 0], axis=0)
                    NewBounds[k, :, 1] = np.min(Bounds[np.logical_not(IncMat[k, :].astype(bool)), k, :, 1], axis=0)
        else:
            NewBounds = np.empty((Bounds.shape[1:]))
            NewBounds[:, 0] = np.max(Bounds[ :, :, 0], axis=0)
            NewBounds[:, 1] = np.min(Bounds[ :, :, 1], axis=0)

    if type(Bounds) is list:
        NewBoundsBuff = np.empty((1, 1, Bounds[0][0].shape[1], 2))
        for k in range(len(Bounds[0])):
            if IncMat in Bounds[1][k]:
                IDSampl = np.where(Bounds[1][k] == IncMat)[0][0]
                NewBoundsBuff = np.append(NewBoundsBuff, Bounds[0][k][IDSampl, :, :].reshape(1, 1, -1, 2), axis=0)
        NewBounds = np.empty((NewBoundsBuff.shape[1:]))
        print(NewBoundsBuff.shape)
        NewBounds[:, :, 0] = np.max(NewBoundsBuff[:, :, :, 0], axis=0)
        NewBounds[:, :, 1] = np.min(NewBoundsBuff[:, :, :, 1], axis=0)
        breakpoint()
    return NewBounds


def PlotRoutine(X, Y, Type, Title, XAxisLabel, YAxisLabel, Legend, Range, fileName):
    """
    :param X: list of abscissas
    :param Y: list of ordinatas
    :param Type: number flag
    :param Title: string
    :param XAxisLabel:
    :param YAxisLabel:
    :param Legend: list of string
    :return: the plots in the browser
    """
    import plotly as plt
    plt.io.renderers.default = "browser"
    fig = plt.graph_objects.Figure()
    if Type == 1:
        fig.add_trace(plt.graph_objects.Scatter(
            x=X,
            y=Y,
            mode='markers',
            name=Legend,
            line_color='blue'
        ))

        fig.update_layout(title_text=Title, title_x=0.5, title_xanchor='center', title_font_size=24,
                          xaxis_title=XAxisLabel, yaxis_title=YAxisLabel, legend_title=Legend, legend_font_size=18)
        fig.update_xaxes(title_font_size=18, tickfont=dict(size=18), range=Range, nticks=9)
        fig.update_yaxes(title_font_size=18, tickfont=dict(size=18), range=Range, nticks=9)
        fig.show()
        fig.write_html(fileName)
    return None


def GriddRFMin(RFModel, BlockVar, NLevels):  # To be completed
    import numpy as np
    import itertools
    ct = 0
    XV = []
    for key in list(BlockVar.keys()):
        if BlockVar[key]['Blocked'] == True:
            XV.append(BlockVar[key]['value'])
        else:
            XV.append(np.linspace(BlockVar[key]['lb'], BlockVar[key]['ub'], NLevels))
        ct = +1
    ct = 0
    full_factorial_combinations = itertools.product(XV)
    XVect = []
    SolRF = []
    for kComb in full_factorial_combinations:
        XVect.append(kComb)
        breakpoint()
        SolRF.append(RFModel.predict(kComb))
        ct = +1
        print(ct)
    return SolRF, XVect

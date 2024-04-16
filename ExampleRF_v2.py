# Generate a dataset 2 features & 1 target

import numpy as np
import pandas as pd
from numpy import ndarray

from bench_fun import *
from sklearn.ensemble import RandomForestRegressor as RfR
import sklearn
from RFOptimLib import *
from matplotlib import pyplot as plt
import pickle
from RFOptimLib import *
import pickle
import numpy as np
import pandas as pd
import plotly as plt
import plotly.graph_objects as go
import math
import time
plt.io.renderers.default = "browser"

from scipy.optimize import milp as milp  # Function to perform the optimization
from scipy.optimize import Bounds as Bounds
from scipy.optimize import LinearConstraint as LinearConstraint
from scipy.sparse import csr_matrix

lbX=np.array([-100,-100]) # lower bounds
ubX=np.array([100,100]) # upper bounds
n_int=20 # cardinality

X1V=np.linspace(lbX[0], ubX[0], num=n_int, endpoint=True) # divisions for feature X1
X2V=np.linspace(lbX[1], ubX[1], num=n_int, endpoint=True) # divisions for feature X2
X1Vfine=np.linspace(lbX[0], ubX[0], num=n_int*50, endpoint=True) # divisions for feature X1 fine
X2Vfine=np.linspace(lbX[1], ubX[1], num=n_int*50, endpoint=True) # divisions for feature X2 fine
Ymesh_fine=tripod(X1Vfine,X2Vfine) # Computation of the function for the fine mesh
Ymesh=tripod(X1V,X2V) # Computation of the function for the coarse mesh
Ylin=Ymesh.reshape((np.prod(Ymesh.shape))) # Linear mesh sample
X1mesh, X2mesh = np.meshgrid(X1V, X2V) # Coarse mesh samples
X1mesh_fine, X2mesh_fine = np.meshgrid(X1Vfine, X2Vfine)  # Fine mesh samples
X1lin=X1mesh.reshape((np.prod(X1mesh.shape))) # Linear mesh sample X1
X2lin=X2mesh.reshape((np.prod(X2mesh.shape))) # Linear mesh sample X2
X1lin_fine=X1mesh_fine.reshape((np.prod(X1mesh_fine.shape))) # Linear fine mesh sample X1
X2lin_fine=X2mesh_fine.reshape((np.prod(X2mesh_fine.shape))) # Linear fine mesh sample X2
data = {'X1': X1lin, 'X2': X2lin,'Y': Ylin}
df = pd.DataFrame.from_dict(data)
Dff=1
N_RF=100 # Number of Random Forest
RndRFModel=[]
if 1: # Flag to train a new RF (1) or load the previous one (0)
    for k in range(N_RF):
        RndRFModel.append(RfR(n_estimators=25, criterion='squared_error', min_samples_split=2, max_features=10,
                         oob_score=True, min_samples_leaf=1, bootstrap=True, verbose=0))
        RndRFModel[-1].fit(df[['X1' ,'X2']], df['Y'])
        root_save_name = 'Tripod_RF_'+str(k)+'.rf'
        pickle.dump(RndRFModel[-1], open(root_save_name, 'wb'))
    pickle.dump(df, open(root_save_name[:-3] + '_dataset', 'wb'))
else:
    root_save_name = 'Tripod_RF.rf'
    RndRFModel = pickle.load(open(root_save_name, 'rb'))

# Reformatting training set
train_In=np.concatenate((df['X1'].values.reshape(-1,1), df['X2'].values.reshape(-1,1)),axis=1)
trainOut=df['Y'].values
N_features=RndRFModel[0][0].tree_.n_features
Feature_Names = ['X1', 'X2']
BlockVar={"X1":                          {"value":0.5, "units":'mm', "lb":-100, "ub":100, "Blocked":False},
          "X2":                          {"value":0.5, "units":'deg', "lb":-100, "ub":100, "Blocked":False},
          }

Constr={'feats':[], 'vals':[]} # Constraints definition (feat: feature number, vals: feature value)

# Optimization with MILP
lbX=np.array([-100,-100]) # To constrain with equality, use lb=ub according to previous constraints
ubX=np.array([100,100])
Res=[]
ResTest=[]

for k in range(N_RF):
    print(k)
    Res.append(OptimRFMILP(RndRFModel[k],lbX, ubX))
    ResTest.append(pd.DataFrame({"X1": [Res[-1].x[0]],"X2": [Res[-1].x[1]]}))

# Verification [till here]
print(RndRFModel.predict(ResTest))

# Gridding Test
GridMin_List=[]
GridMinArg_List=[]
Dff_List=[]
import time
n_intV=range(10,1000,10)
grd=np.empty(len(n_intV))
grdT=np.empty(len(n_intV))
grdArg=np.empty(len(n_intV))
for k in range(N_RF):
    ct=0
    for n_int in n_intV:
        start_time = time.time()
        X1V=np.linspace(lbX[0], ubX[0], num=n_int, endpoint=True)
        X2V=np.linspace(lbX[1], ubX[1], num=n_int, endpoint=True)
        Xv, Yv = np.meshgrid(X1V, X2V)
        Xv = Xv.reshape((np.prod(Xv.shape)))
        Yv = Yv.reshape((np.prod(Yv.shape)))
        grd[ct]=RndRFModel[k].predict(pd.DataFrame({"X1": Xv,"X2": Yv})).min()
        grdArg[ct]=RndRFModel[k].predict(pd.DataFrame({"X1": Xv,"X2": Yv})).argmin()
        grdT[ct]=time.time()-start_time
        print("execution time in [s]: "+str(grdT[ct]))
        ct+=1
    print(grd[-1])
    GridMin_List.append(grd[-1])
    SolGrid=[Xv[int(grdArg[-1])], Yv[int(grdArg[-1])]]
    GridMinArg_List.append(SolGrid)
    Dff=max(abs(Res[k].fun-RndRFModel[k].predict(ResTest[k])),abs(Res[k].fun-grd[-1].min()),abs(RndRFModel[k].predict(ResTest[k])-grd[-1].min()))
    print(Dff)
    Dff_List.append(Dff)
'''
plt.io.renderers.default = "browser"
xplt=np.linspace(lbX[0], ubX[0], num=1000, endpoint=True)*0+Constr['vals'][0]
yplt=np.linspace(lbX[1], ubX[1], num=1000, endpoint=True)
zplt=tripod(xplt,yplt)
fig=plt.graph_objects.Figure()
fig.add_trace(go.Scatter(
                         x=yplt,
                         y=zplt[:,0],
                         mode='lines',
                         marker=dict(size=3)))
fig.update_layout(title="Function section (X1="+str(Constr['vals'][0])+")",title_x=0.5, xaxis_title="X2", yaxis_title="f")
figMins.update_xaxes(title_font_size=18, tickfont=dict(size=18))
figMins.update_yaxes(title_font_size=18, tickfont=dict(size=18))
fig.show()
fig.write_html("Section_function_"+str(time.time())[:10]+".html")
'''
# Definition of the Incidence matrix relating each sample to the tree that have been trained with it
toll=abs(np.diff(trainOut)).min()
decs=int(np.round(math.log10(1/toll))) # Definition of the round to not create fake distinction among minima
TTIncidenceM_List=[]

for k in range(N_RF):
    TTIncidenceM = np.empty((train_In.shape[0], RndRFModel[k].n_estimators))
    for u in range(RndRFModel[k].n_estimators):
        TTIncidenceM[:,u]=abs(RndRFModel[k][u].predict(train_In)-trainOut.reshape(1,-1))>toll # One if the sample is MISSING in the training set
    TTIncidenceM_List.append(TTIncidenceM)

#TTIncidenceM=np.zeros(TTIncidenceM.shape) # For debug
'''
plt.io.renderers.default = "browser"
fig = go.Figure(go.Scatter3d(x=df['X1'].values, y=df['X2'], z=df['Y'],mode='markers',marker=dict(size=3)))
fig.update_scenes(xaxis_title="X1", yaxis_title="X2", zaxis_title="Y")
fig.update_layout(title="Dataset",title_x=0.5)
fig.show()
fig.write_html("Dataset.html")

fig1 = go.Figure(go.Surface(x=X1Vfine, y=X2Vfine, z=tripod(X1Vfine,X2Vfine))
fig1.update_scenes(xaxis_title="X1", yaxis_title="X2", zaxis_title="Y")
fig1.update_layout(title="Tripod function",title_x=0.5)
fig1.show()
fig1.write_html("Tripod_function_"+str(time.time())[:10]+".html")
fig1.show()

fig2 = go.Figure(go.Scatter(x=np.array(n_intV), y=grd))
fig2.update_layout(xaxis_title="n_steps", yaxis_title="min values")
fig2.update_layout(title="Gridding convergence",title_x=0.5)
fig2.show()
fig3 = go.Figure(go.Scatter(x=np.array(n_intV), y=grdT))
fig3.update_layout(xaxis_title="n_steps", yaxis_title="time")
fig3.update_layout(title="Gridding convergence",title_x=0.5)
fig3.show()

# Plot for NT=6
from plotly.subplots import make_subplots
tls=[]
for k in range(len(RndRFModel)):
    tls.append('Tr. n.'+str(k))

fig4=plt.subplots.make_subplots(rows=2,cols=3,specs=[[{'is_3d': True}, {'is_3d': True},{'is_3d': True}],
                                                     [{'is_3d': True}, {'is_3d': True},{'is_3d': True}]],
                                subplot_titles=tls)
ct=0
for ii in range(2):
    for jj in range(3):
        yRF=RndRFModel[ct].predict(np.concatenate((X1lin_fine.reshape(-1,1), X2lin_fine.reshape(-1,1)),axis=1))
        PointsIn = train_In[TTIncidenceM[:, ct] == 0, :]
        PointsOut = trainOut[TTIncidenceM[:, ct] == 0]
        ct+=1
        fig4.add_trace(go.Surface(x=X1Vfine, y=X2Vfine, z=yRF.reshape(len(X1Vfine),len(X2Vfine)),colorbar=None,showscale=False),row=ii+1,col=jj+1)
        fig4.update_scenes(xaxis_title="X1", yaxis_title="X2", zaxis_title="Y",xaxis_visible=False,yaxis_visible=False,zaxis_visible=False)
        # Adding points seen during training
        fig4.add_trace(go.Scatter3d(x=PointsIn[:,0], y=PointsIn[:,1], z=PointsOut, mode='markers',marker=dict(size=6, color='black',symbol='cross')),row=ii+1,col=jj+1)
fig4.update_layout(title="RF Trees (first 6)", title_x=0.5,showlegend=False)
fig4.write_html("RF_Trees.html")
fig4.show()

# Random Forest
yRF=RndRFModel.predict(np.concatenate((X1lin_fine.reshape(-1,1), X2lin_fine.reshape(-1,1)),axis=1))
fig5=go.Figure(go.Surface(x=X1Vfine, y=X2Vfine, z=yRF.reshape(len(X1Vfine),len(X2Vfine)),colorbar=None))
fig5.update_scenes(xaxis_title="X1", yaxis_title="X2", zaxis_title="Y")
fig5.update_layout(title="RF complete", title_x=0.5)
fig5.write_html("RF_Complete_"+str(time.time())[:10]+".html")
fig5.show()

# Random Forest (Section)
plt.io.renderers.default = "browser"
zplt=RndRFModel.predict(np.concatenate((X1lin_fine.reshape(-1,1)*0+5, X2lin_fine.reshape(-1,1)),axis=1))
fig=plt.graph_objects.Figure()
fig.add_trace(go.Scatter(
                         x=X2lin_fine,
                         y=zplt,
                         mode='lines',
                         marker=dict(size=3)))
fig.update_layout(title="RF section (X1="+str(Constr['vals'][0])+")",title_x=0.5, xaxis_title="X2", yaxis_title="f")
figMins.update_xaxes(title_font_size=18, tickfont=dict(size=18))
figMins.update_yaxes(title_font_size=18, tickfont=dict(size=18))
fig.show()
fig.write_html("Section_RF_"+str(time.time())[:10]+".html")
'''

# New approach
# No rescaling. Variables defined for formal reasons
NNFacts={}
NNFacts['InMean']=np.array([0,0])
NNFacts['InStd']=np.array([1,1])
NNFacts['OutMean']=np.array([0])
NNFacts['OutStd']=np.array([1])

CentralSolMinArg_List=[]
CentralSolPROJMinArg_List=[]
CentralSolMinVal_List=[]
CentralSolPROJMinVal_List=[]
TTIncidenceMGlob_List=[]

for k in range(N_RF):
    Bounds=FindPartitions(RndRFModel[k],train_In,BlockVar,NNFacts) # Partitions computation
    Compat=FindPartitionsComp(Bounds, Constr) # Partitions compatible (1 if the partition is compatible)
    TTIncidenceMGlob=np.logical_not((TTIncidenceM_List[k])).astype(int)*Compat.transpose() #
    TTIncidenceMGlob_List.append(TTIncidenceMGlob)
    print(k)

    # Identification of the samples associatedd to compatible set
    IdCompSampl=np.where(TTIncidenceMGlob.sum(axis=1).astype(bool)==True)[0]

    # Identification of the best point in the data set
    IdSort=np.argsort(trainOut[IdCompSampl].reshape(-1),axis=0) # Sorting only the samples that have been considered in at least one tree
    IdMinDset=IdCompSampl[IdSort[0]] # Taking the absolute minimum
    CompatBoundsMin=Bounds[Compat[:,IdMinDset].astype(bool),IdMinDset,:,:] # Partitions associated to the absolute Minimum
    OverallBounds=InterSectBounds(CompatBoundsMin,None) # There should be at least one etree that have seen one samples
    CentralSolMinArg=pd.DataFrame({"X1": [OverallBounds.mean(axis=1)[0]],"X2": [OverallBounds.mean(axis=1)[1]]})
    CentralSolMinArg_List.append(CentralSolMinArg) # Central solution NOT Projected onto constraints
    CentralSolMinVal_List.append(RndRFModel[k].predict(CentralSolMinArg)) # Central solution value computed by the RF
    CentralSolPROJ=CentralSolMinArg
    CentralSolPROJ[CentralSolPROJ.keys()[Constr['feats']]]=Constr['vals']
    CentralSolPROJMinArg_List.append(CentralSolPROJ)
    CentralSolMinVal_List.append(RndRFModel[k].predict(CentralSolPROJ))

# !!! Sorting Constrained !!!
#Initialization of Lists for Montecarlo
MinFunVals_List=[] # RF evaluation at the input vector corresponding to the absolute minimum target in training set
MinTarg_List=[] # Values of the absolute minimum targets in training set
MinX_List=[] # Inputs associated to values of the absolute minimum targets in training set
unVals_List=[] # Values of the absolute minima targets in training set for the various trees
RecId_List=[] # Id of the trees corresponding to values of the absolute minima targets in training set for the various trees
unCounts_List=[] # Number of trees with the same values of the absolute minima targets in training set for the various trees
CentralSol_List=[] # Cental solutions, only the absolute minimum
CentralSolPROJ_List=[] # Central solution projected, only the absolute minimum
CentralSolSort_List=[] # Cental solutions, for all the minima
CentralSolSortPROJ_List=[] # Central solution projected, for all the minima
unValsFun_List=[] # List of vectors containing the RF evaluation at the argument of the minima of training set
unValsFunCentr_List=[] # List of vectors containing the RF evaluation at the central point of the intersection of partitions associated to the minima of training set
unValsFunCentrPROJ_List=[] # List of vectors containing the RF evaluation at the central point of the intersection of partitions associated to the minima of training set, projected to the constraints
MinXSort_List=[] # List of argument of the minima of all trees

for w in range(N_RF):
    # Initializations
    IdSortOrigIndex=np.empty(len(RndRFModel[w]),dtype=int)
    MinTarg=np.empty(len(RndRFModel[w]))
    MinX=np.empty((len(RndRFModel[w]),2))
    IdSamplesUsed=[]
    for k in range(len(RndRFModel[w])): # For all trees in the Forest
        IdSamplesUsed.append(np.where(TTIncidenceMGlob_List[w][:,k]==1)[0]) # Indeces of samples used to train the k-th tree
        IdSort=np.argmin(trainOut[IdSamplesUsed[-1]]) # Find the minimum among the used samples
        IdSortOrigIndex[k]=IdSamplesUsed[-1][IdSort] # Get the indeces w.r.t. the original dataset
        MinTarg[k]=trainOut[IdSortOrigIndex[k]] # Value of the minimum (target)
        MinX[k,:]=train_In[IdSortOrigIndex[k],:] # Input corresponding to the minimum
        MinFunVals=RndRFModel[w].predict(pd.DataFrame({"X1": MinX[:,0],"X2": MinX[:,1]})) # Value predicted by RF at the minimum target input
        MinFunVals_List.append(MinFunVals) # append to the list used to iterate for Montecarlo
        MinTarg_List.append(MinTarg[k]) # append to the list used to iterate for Montecarlo
        MinX_List.append(MinX[k,:]) # append to the list used to iterate for Montecarlo
    unVals, RecId, unCounts=np.unique(MinTarg.round(decimals=decs-1), return_counts=True, return_inverse=True) # determine the number of distinct minima and corresponding Trees
    unVals_List.append(unVals) # append to the list used to iterate for Montecarlo
    RecId_List.append(RecId) # append to the list used to iterate for Montecarlo
    unCounts_List.append(unCounts) # append to the list used to iterate for Montecarlo

    # Computation of the central value for each minima
    CentralSol=[] # Intialization to void list
    CentralSolPROJ=[] # Intialization to void list

    for k in range(len(unVals)):
        IdVal = np.where(RecId == k) # identify the trees trained with a given minimum
        IdMinDset=IdSortOrigIndex[IdVal[0][0]] # identification of the index of the k-th minimum in the dataset
        CompatBoundsMin = Bounds[Compat[:, IdMinDset].astype(bool), IdMinDset, :, :] # identification of the compatible partitions associated to the k-th minimum
        OverallBounds = InterSectBounds(CompatBoundsMin, None) # intersect the candidate partitions associated to the k-th minimum
        CentralSol.append(pd.DataFrame({"X1": [OverallBounds.mean(axis=1)[0]], "X2": [OverallBounds.mean(axis=1)[1]]})) # append the central solution
        CentralSolPROJ.append(pd.DataFrame({"X1": [OverallBounds.mean(axis=1)[0]], "X2": [OverallBounds.mean(axis=1)[1]]})) # append the central solution
        CentralSolPROJ[-1][CentralSolPROJ[-1].keys()[Constr['feats']]]=Constr['vals'] # Project the central solution

    unValsFun=np.empty_like(unVals) # Initializing the vector containing the function evaluation where there are minima
    unValsFunCentr=np.empty_like(unVals)
    unValsFunCentrPROJ=np.empty_like(unVals)
    InpVectMin=[]
    for k in range(len(unVals)):
        IdVal=np.where(RecId==k)
        unValsFun[k]=RndRFModel[w].predict(pd.DataFrame({"X1": [train_In[IdSortOrigIndex[IdVal[0][0]],0]],"X2": [train_In[IdSortOrigIndex[IdVal[0][0]],1]]}))
        InpVectMin.append(train_In[IdSortOrigIndex[IdVal[0][0]],:])
        unValsFunCentr[k]=RndRFModel[w].predict(CentralSol[k])
        unValsFunCentrPROJ[k]=RndRFModel[w].predict(CentralSolPROJ[k])

    CentralSolSort_List.append(CentralSol)
    CentralSolSortPROJ_List.append(CentralSolPROJ)
    unValsFun_List.append(unValsFun)
    unValsFunCentr_List.append(unValsFunCentr)
    unValsFunCentrPROJ_List.append(unValsFunCentrPROJ)
    MinXSort_List.append(InpVectMin)

# Find the best of the bests
N_of_Mins=5
SolHeurVal=[]
SolHeurX=[]
SolHeurXArray=np.empty((0,2))
SolMilp=[]

for w in range(N_RF):
    SolHeurVal.append(unValsFunCentrPROJ_List[w].min())
    SolHeurX.append(CentralSolSortPROJ_List[w][unValsFunCentrPROJ_List[w].argmin()])
    SolHeurXArray=np.append(SolHeurXArray,SolHeurX[w].values,axis=0)
    SolMilp.append(Res[w].fun)

SubOpt_Average=(np.array(SolHeurVal)-np.array(GridMin_List)).mean()
SubOpt_Std=(np.array(SolHeurVal)-np.array(GridMin_List)).std()

valsXHeu, IdsXHeu, countsXHeu=np.unique(SolHeurXArray,return_counts=True, return_inverse=True,axis=0)
valsXGrid, IdsXGrid, countsXGrid=np.unique(np.array(GridMinArg_List),return_counts=True, return_inverse=True,axis=0)

# Plotting everything

figSolsVal=plt.graph_objects.Figure()
figSolsVal.add_trace(
plt.graph_objects.Scatter(
            x = list(range(1,N_RF+1)),
            y = SolHeurVal, #tuple(range(1,len(unVals)+1)),
            mode = 'markers+lines',
            name = 'Heuristic solution',
            line_color = 'green',
            marker_size=5
            ))
figSolsVal.add_trace(
plt.graph_objects.Scatter(
            x = list(range(1,N_RF+1)),
            y = GridMin_List, #tuple(range(1,len(unVals)+1)),
            mode = 'markers+lines',
            name = 'Absolute minimum',
            line_color = 'red',
            marker_size=5
            ))
figSolsVal.update_layout(title_text='Monte Carlo analysis - Exact vs Heuristic solution', title_x=0.5, title_xanchor='center',
                        title_font_size=24, yaxis_title='min. value', xaxis_title="# RF",
                        legend_title="Legend", legend_font_size=18)
figSolsVal.update_xaxes(title_font_size=18, tickfont=dict(size=18))
figSolsVal.update_yaxes(title_font_size=18, tickfont=dict(size=18))
figSolsVal.write_html("Montecarlo_analysis_"+str(time.time())[:10]+".html")
figSolsVal.show()

# Relative error
from plotly.subplots import make_subplots
figSolsVal2=make_subplots(specs=[[{"secondary_y": True}]])
figSolsVal2.add_trace(
plt.graph_objects.Scatter(
            x = list(range(1,N_RF+1)),
            y = (np.array(SolHeurVal)-np.array(GridMin_List))/np.array(GridMin_List)*100, #tuple(range(1,len(unVals)+1)),
            mode = 'markers+lines',
            name = 'Relative error [%]',
            line_color = 'red',
            marker_size=11
            ),secondary_y=False)
figSolsVal2.add_trace(
plt.graph_objects.Scatter(
            x = list(range(1,N_RF+1)),
            y = (np.array(SolHeurVal)-np.array(GridMin_List)), #tuple(range(1,len(unVals)+1)),
            mode = 'markers+lines',
            name = 'absolute error',
            line_color = 'green',
            marker_size=11
            ),secondary_y=True)
figSolsVal2.update_layout(title_text='Monte Carlo analysis - Exact vs Heuristic solution (function value)', title_x=0.5, title_xanchor='center',
                        title_font_size=24, xaxis_title="# RF", legend_font_size=24)

figSolsVal2.update_xaxes(title_font_size=24, tickfont=dict(size=24))
figSolsVal2.update_yaxes(title_font_size=24, tickfont=dict(size=24), title_text='Absolute error', secondary_y=True)
figSolsVal2.update_yaxes(title_font_size=24, tickfont=dict(size=24), title_text='Relative error [%]', secondary_y=False)
figSolsVal2.write_html("Montecarlo_analysis_"+str(time.time())[:10]+".html")
figSolsVal2.show()

IdXErr=np.where((np.linalg.norm(SolHeurXArray-np.array(GridMinArg_List), ord=2,axis=1))>
                (np.linalg.norm(SolHeurXArray-np.array(GridMinArg_List), ord=2,axis=1).mean()+
                 np.linalg.norm(SolHeurXArray-np.array(GridMinArg_List), ord=2,axis=1).std()))
IdXCoinc=np.delete(np.array(range(len(SolHeurXArray))), np.intersect1d(np.array(range(len(SolHeurXArray))),IdXErr))

valsXHeuSim, IdsXHeuSim, countsXHeuSim=np.unique(SolHeurXArray[IdXCoinc],return_counts=True, return_inverse=True,axis=0)
valsXHeuDiff, IdsXHeuDiff, countsXHeuDiff=np.unique(SolHeurXArray[IdXErr],return_counts=True, return_inverse=True,axis=0)

figSolsX=plt.graph_objects.Figure()
figSolsX.add_trace(
plt.graph_objects.Scatter(
            x = valsXHeuSim[:,0],
            y = valsXHeuSim[:,1], #tuple(range(1,len(unals)V+1)),
            mode = 'markers',
            name = 'Approx. nearly exact',
            marker_color = 'green',
            marker_line_color='black',
            marker_line_width=2,
            marker_size=countsXHeuSim*2
            ))

figSolsX.add_trace(
plt.graph_objects.Scatter(
            x=valsXHeuDiff[:, 0],
            y=valsXHeuDiff[:, 1],  # tuple(range(1,len(unals)V+1)),
            mode = 'markers',
            name = 'Approx. largely different',
            marker_color = 'yellow',
            marker_line_color='black',
            marker_line_width=2,
            marker_size=countsXHeuDiff*2
            ))

figSolsX.add_trace(
plt.graph_objects.Scatter(
            x = valsXGrid[:,0],
            y = valsXGrid[:,1], #tuple(range(1,len(unVals)+1)),
            mode = 'markers',
            name = 'Exact RF minimum',
            marker_color = 'red',
            marker_line_color='black',
            marker_line_width=2,
            marker_size=countsXGrid*2
            ))
figSolsX.add_trace(
plt.graph_objects.Scatter(
            x = [-47.4],
            y = [47.4], #tuple(range(1,len(unVals)+1)),
            mode = 'markers',
            name = 'Dataset minimum',
            marker_color = 'black',
            marker_symbol='cross',
            marker_line_color='black',
            marker_line_width=2,
            marker_size=countsXGrid
            ))
figSolsX.update_layout(title_text='Approximated vs exact minimizers', title_x=0.5, title_xanchor='center',
                        title_font_size=24, yaxis_title='X2', xaxis_title="X1",
                        legend_title="", legend_font_size=24)
figSolsX.update_xaxes(title_font_size=24, tickfont=dict(size=24))
figSolsX.update_yaxes(title_font_size=24, tickfont=dict(size=24))
figSolsX.write_html("Montecarlo_analysis_X"+str(time.time())[:10]+".html")
figSolsX.show()

# Plot of distances

figSolsVal3=make_subplots(specs=[[{"secondary_y": True}]])
figSolsVal3.add_trace(
plt.graph_objects.Scatter(
            x = list(range(1,N_RF+1)),
            y = np.linalg.norm(SolHeurXArray-np.array(GridMinArg_List), ord=2,axis=1), #tuple(range(1,len(unVals)+1)),
            mode = 'markers+lines',
            name = 'Distance between solutions',
            line_color = 'black',
            marker_size=11
            ),secondary_y=False)
figSolsVal3.update_layout(title_text='Monte Carlo analysis - Exact vs Heuristic solution (argument)', title_x=0.5, title_xanchor='center',
                        title_font_size=24, yaxis_title='$\LARGE{||\mathbf{x}^*_{exact}-\mathbf{x}^*_{heuristic}||_2}$', xaxis_title="# RF")
figSolsVal3.update_xaxes(title_font_size=24, tickfont=dict(size=24))
figSolsVal3.update_yaxes(title_font_size=24, tickfont=dict(size=24))
figSolsVal3.write_html("Montecarlo_analysis_"+str(time.time())[:10]+".html")
figSolsVal3.show()

figMins3=plt.graph_objects.Figure()
figMins3.add_trace(
plt.graph_objects.Scatter(
            x = unVals_List[0],
            y = unCounts_List[0], #tuple(range(1,len(unVals)+1)),
            mode = 'markers',
            name = r'$\LARGE y^*_k \in \mathcal{D}_j   |$',
            line_color = 'red',
            marker_size=15
            ))
figMins3.add_trace(
plt.graph_objects.Scatter(
            x = unValsFun_List[0],
            y = unCounts_List[0],
            mode = 'markers',
            name = r'\par\par\par $\LARGE R(\mathbf{x}_k^*)$',
            line_color = 'green',
            marker_size=15
            ))

figMins3.update_layout(title_text='Analysis of the best training targets', title_x=0.5, title_xanchor='center',
                        title_font_size=30, yaxis_title=r"$\LARGE \text{N.  of  RTs  with  } (\mathbf{x}_k^*,y^*_k)\text{  in  }\mathcal{D}_j$", xaxis_title=r'$\LARGE y^*_k, R(\mathbf{x}_k^*) \text{ values}$',
                        legend_title='',legend_font_size=18)
figMins3.update_xaxes(title_font_size=24, tickfont=dict(size=24))
figMins3.update_yaxes(title_font_size=24, tickfont=dict(size=24))
figMins3.write_html("MinTargs_ex_"+str(time.time())[:10]+".html")
figMins3.show()
oob_error=(RndRFModel[0].oob_prediction_-trainOut.reshape(-1))

figHis=plt.graph_objects.Figure()
figHis.add_trace(plt.graph_objects.Histogram(x=oob_error, name="OOB Errors distributions",histnorm='probability'))
figHis.update_layout(title_text='OOB prediction error (empirical distribution for RF #1)',title_x=0.5,title_xanchor='center',title_font_size=24, xaxis_title='error value', yaxis_title="Probability",legend_title="Legend", legend_font_size=18)
figHis.update_xaxes(title_font_size=24, tickfont=dict(size=24))
figHis.update_yaxes(title_font_size=24, tickfont=dict(size=24))
figHis.show()

from plotly.subplots import make_subplots
figHis2=make_subplots(rows=2, cols=1)
figHis2.add_trace(plt.graph_objects.Histogram(x=(np.array(SolHeurVal)-np.array(GridMin_List))/np.array(GridMin_List)*100, name="rel. error",histnorm=''), row=1, col=1)
figHis2.add_trace(plt.graph_objects.Histogram(x=(np.array(SolHeurVal)-np.array(GridMin_List)), name="abs. error",histnorm=''), row=2, col=1)
figHis2.update_layout(title_text="Minimun value error distribution - Exact vs Approx. for 100 RFs",title_x=0.5,title_xanchor='center',title_font_size=24, legend_title="", legend_font_size=24)
figHis2.update_xaxes(title_font_size=24, tickfont=dict(size=24), title_text='relative error [%]', row=1,col=1)
figHis2.update_yaxes(title_font_size=24, tickfont=dict(size=24), title_text='count', row=1,col=1)
figHis2.update_xaxes(title_font_size=24, tickfont=dict(size=24), title_text='absolute error', row=2,col=1)
figHis2.update_yaxes(title_font_size=24, tickfont=dict(size=24), title_text='count', row=2,col=1)
figHis2.show()

#----------------------
figMins=plt.graph_objects.Figure()
figMins.add_trace(
plt.graph_objects.Scatter(
            x = unVals_List[0],
            y = unCounts_List[0], #tuple(range(1,len(unVals)+1)),
            mode = 'markers',
            name = 'Minimum targets (y*_k)',
            line_color = 'red',
            marker_size=10
            ))
figMins.add_trace(
plt.graph_objects.Scatter(
            x = unValsFun_List[0],
            y = unCounts_List[0],
            mode = 'markers',
            name = 'f(x*_k)',
            line_color = 'green',
            marker_size=10))

figMins.add_trace(
plt.graph_objects.Scatter(
            x = unValsFunCentr_List[0],
            y = unCounts_List[0],
            mode = 'markers',
            name = 'f(Intersection centre)',
            line_color = 'yellow',
            marker_size=10))

figMins.add_trace(
plt.graph_objects.Scatter(
            x = unValsFunCentrPROJ_List[0],
            y = unCounts_List[0],
            mode = 'markers',
            name = 'f(Intersection centre Projected)',
            line_color = 'black',
            marker_size=10))

figMins.update_layout(title_text='Best training targets and corresponding RF predictions', title_x=0.5, title_xanchor='center',
                        title_font_size=24, yaxis_title='Trees count with for sample', xaxis_title="Out value",
                        legend_title="Legend", legend_font_size=18)
figMins.update_xaxes(title_font_size=18, tickfont=dict(size=18))
figMins.update_yaxes(title_font_size=18, tickfont=dict(size=18))
figMins.write_html("MinTargs_ex_"+str(time.time())[:10]+".html")
figMins.show()

# Random Forest
yRF=RndRFModel.predict(np.concatenate((X1lin_fine.reshape(-1,1), X2lin_fine.reshape(-1,1)),axis=1))
ct+=1
figMins2=plt.graph_objects.Figure()
figMins2=go.Figure(go.Surface(x=X1Vfine, y=X2Vfine, z=yRF.reshape(len(X1Vfine),len(X2Vfine)),colorbar=None,opacity=0.5))
figMins2.add_trace(go.Scatter3d(x=MinX[:,0], y=MinX[:,1], z=MinTarg, mode='markers',marker=dict(size=6, color='blue')))
figMins2.add_trace(go.Scatter3d(x=np.array(CentralSolPROJ).reshape(-1,2)[:,0],
                                y=np.array(CentralSolPROJ).reshape(-1,2)[:,1],
                                z=np.array(unValsFunCentrPROJ), mode='markers',marker=dict(size=6, color='black')))
figMins2.add_trace(go.Scatter3d(x=[SolGrid[0]],
                                y=[SolGrid[1]],
                                z=[grd[-1]], mode='markers',marker=dict(size=6, color='green')))
figMins2.update_scenes(xaxis_title="X1", yaxis_title="X2", zaxis_title="Y")
figMins2.update_layout(title="RF complete", title_x=0.5)
figMins2.write_html("RF_Complete_with_samples_"+str(time.time())[:10]+".html")
figMins2.show()

# Sorting NOT Constrained
IdSortOrigIndex=np.empty(len(RndRFModel),dtype=int)
MinTarg=np.empty(len(RndRFModel))
MinX=np.empty((len(RndRFModel),2))
IdSamplesUsed=[]
for k in range(len(RndRFModel)): # For all trees in the Forest
    IdSamplesUsed.append(np.where(TTIncidenceM[:,k]==0)[0])
    IdSort=np.argmin(trainOut[IdSamplesUsed[-1]])
    IdSortOrigIndex[k]=IdSamplesUsed[-1][IdSort]
    MinTarg[k]=trainOut[IdSortOrigIndex[k]]
    MinX[k,:]=train_In[IdSortOrigIndex[k],:]
MinFunVals=RndRFModel.predict(pd.DataFrame({"X1": MinX[:,0],"X2": MinX[:,1]}))

unVals, RecId, unCounts=np.unique(MinTarg, return_counts=True, return_inverse=True)
unValsFun=np.empty_like(unVals)
for k in range(len(unVals)):
    IdVal=np.where(RecId==k)
    unValsFun[k]=RndRFModel.predict(pd.DataFrame({"X1": [train_In[IdSortOrigIndex[IdVal[0][0]],0]],"X2": [train_In[IdSortOrigIndex[IdVal[0][0]],1]]}))

figMins=plt.graph_objects.Figure()
figMins.add_trace(
plt.graph_objects.Scatter(
            x = unVals,
            y = unCounts, #tuple(range(1,len(unVals)+1)),
            mode = 'markers',
            name = 'Minimum targets (y*_k)',
            line_color = 'red',
            marker_size=10
            ))
figMins.add_trace(
plt.graph_objects.Scatter(
            x = unValsFun,
            y = unCounts,
            mode = 'markers',
            name = 'f(x*_k)',
            line_color = 'green',
            marker_size=10))

figMins.update_layout(title_text='Best training targets and corresponding RF predictions', title_x=0.5, title_xanchor='center',
                        title_font_size=24, yaxis_title='Trees count with for sample', xaxis_title="Out value",
                        legend_title="Legend", legend_font_size=18)
figMins.update_xaxes(title_font_size=18, tickfont=dict(size=18))
figMins.update_yaxes(title_font_size=18, tickfont=dict(size=18))
figMins.write_html("MinTargs_ex_"+str(time.time())[:10]+".html")
figMins.show()

# Random Forest
yRF=RndRFModel.predict(np.concatenate((X1lin_fine.reshape(-1,1), X2lin_fine.reshape(-1,1)),axis=1))
ct+=1
figMins2=plt.graph_objects.Figure()
figMins2=go.Figure(go.Surface(x=X1Vfine, y=X2Vfine, z=yRF.reshape(len(X1Vfine),len(X2Vfine)),colorbar=None,opacity=0.5))
figMins2.add_trace(go.Scatter3d(x=MinX[:,0], y=MinX[:,1], z=MinTarg, mode='markers',marker=dict(size=6, color='blue')))
figMins2.add_trace(go.Scatter3d(x=MinX[:,0], y=MinX[:,1], z=MinFunVals, mode='markers',marker=dict(size=6, color='black')))
figMins2.add_trace(go.Scatter3d(x=ResTest['X1'], y=ResTest['X2'], z=RndRFModel.predict(ResTest), mode='markers',marker=dict(size=6, color='green')))
figMins2.update_scenes(xaxis_title="X1", yaxis_title="X2", zaxis_title="Y")
figMins2.update_layout(title="RF complete", title_x=0.5)
figMins2.write_html("RF_Complete_with_samples_"+str(time.time())[:10]+".html")
figMins2.show()

xP=np.empty(0)
yP=np.empty(0)
ct=0
for k in range(len(Bounds.shape[0])):
    for j in range((Bounds.shape[1])):
        x_temp: ndarray=np.array((Bounds[k,j,0,0],Bounds[k,j,0,0],Bounds[k,j,0,1],Bounds[k,j,0,1],Bounds[k,j,0,0], None))
        y_temp = np.array((Bounds[k, j, 1, 0], Bounds[k, j, 1, 1], Bounds[k, j, 1, 0], Bounds[k, j, 1, 1], Bounds[k, j, 1, 0], None))
        xP=np.append(xP,x_temp)
        yP = np.append(xP, x_temp)

# Apply constraints
# Plot OOB errors
# Definition of constraints on free variables, in the form of a list of lists (first element feature n, second element

Constr={}
IdSort=[]; IdSortAbs=[]
IdSortAbsComplete=[]
MinTarg=np.empty(RndRFModel.n_estimators)
MinTarg=np.empty(RndRFModel.n_estimators)
MinPred=np.empty(RndRFModel.n_estimators)
BoundsBuff = np.copy(Bounds)
BoundsCompat=[]
BoundsCompatIdx=[]
for k in range(len(RndRFModel)): #range(len(RndRFModel)): # Get the decision Trees
    #Identifiaction of the partitions compatible with the constraints
    IdParts = np.array(range(N_Samples))
    IdPartsComplete = np.array(range(N_Samples))
    CheckPoints=(RndRFModel[k].predict(train_In)!=trainOutRnd.reshape(1,-1))
    IdPartsElim=np.empty((0,1),dtype=int)
    IdPartsElimComplete = np.empty((0, 1), dtype=int)
    for h in range(len(Constr['feats'])):
        WhInds=np.where((BoundsBuff[k, :, Constr['feats'][h], 0] >= Constr['vals'][h]) |
                               (BoundsBuff[k, :,Constr['feats'][h], 1] < Constr['vals'][h]))[1]
        IdPartsElim=np.append(IdPartsElim,WhInds)
    # Eliminating the Ids that have been not used in the
    IdPartsElimComplete=np.copy(IdPartsElim)
    IdPartsComplete=np.delete(IdPartsComplete,IdPartsElimComplete)
    IdPartsElim = np.append(IdPartsElim,np.asarray(CheckPoints==True).nonzero()[1])
    IdParts=np.delete(IdParts,IdPartsElim)
    BoundsCompat.append(np.delete(BoundsBuff[k,:,:,:], IdPartsElimComplete, axis=0))
    BoundsCompatIdx.append(IdPartsComplete)
    # Find the minimum of the targets
    IdSort.append(np.argmin(trainOutRnd[IdParts].reshape(-1), axis=0))
    IdMin = IdSort[k]
    IdSortAbs.append(IdParts[IdMin])
    MinTarg[k]=trainOutRnd[IdParts[IdMin]]*NNFacts['OutStd']+NNFacts['OutMean']
    MinPred[k]=RndRFModel.predict(train_In[IdParts[IdMin],:].reshape(1,-1))*NNFacts['OutStd']+NNFacts['OutMean']
# Create the list for the following intersection
BCStr=[]
BCStr.append(BoundsCompat) # Compatible partitions
BCStr.append(BoundsCompatIdx) # Samples Ids
unVals, RecId, unCounts=np.unique(MinPred, return_counts=True, return_inverse=True)
unValsTarg, RecIdTarg, unCountsTarg=np.unique(MinTarg, return_counts=True, return_inverse=True)
unValsFun=np.empty_like(unVals)
for k in range(len(unValsTarg)):
    IdVal=np.where(RecIdTarg==k)
    unValsFun[k]=unVals[RecId[IdVal[0][0]]]

IdMins=np.where((trainOutRnd*NNFacts['OutStd']+NNFacts['OutMean'])==np.unique(MinTarg))
errVals=abs(oob_error[IdMins[0][np.argsort(IdMins[1])]])
errValsSign=oob_error[IdMins[0][np.argsort(IdMins[1])]]



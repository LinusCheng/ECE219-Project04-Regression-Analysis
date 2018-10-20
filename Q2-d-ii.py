import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn import metrics
#from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
#from sklearn.feature_selection import f_regression, mutual_info_regression
import matplotlib.pyplot as plt
import time




""" ====== Prepare data ====== """

content = pd.read_csv("network_backup_dataset.csv")
content = content.replace({'Day of Week': {'Monday' : 0, 'Tuesday' : 1, 'Wednesday' : 2 ,
                                         'Thursday' : 3, 'Friday' : 4,
                                         'Saturday' : 5, 'Sunday' : 6 }})

WorkFlow_list = sorted(pd.unique(content['Work-Flow-ID']))
File_list     = sorted(pd.unique(content['File Name']))  
num_WorkFlow  = len(WorkFlow_list)
num_File      = len(File_list)

for i in np.arange(num_WorkFlow):
    content = content.replace({'Work-Flow-ID': {'work_flow_%d'%i:i}})

for i in np.arange(num_File):
    content = content.replace({'File Name': {'File_%d'%i:i}})


cont_np = content.values

X_list = []
Y_list = []

for i in np.arange(5):
    cont_list_i = cont_np[np.where(cont_np[:,3] == i)]
    X_i = np.delete(cont_list_i[: , 0:5],3,axis=1)
    Y_i = cont_list_i[: ,  5 ]
    X_list.append(X_i)
    Y_list.append(Y_i)




del WorkFlow_list,File_list,cont_np, num_WorkFlow,num_File,i,X_i,Y_i,cont_list_i
# delete unused variables






def kfold_ridg(X,Y,alpha):
    test_rmse  = []
    train_rmse = []
    kf = KFold(n_splits=10, random_state=None, shuffle=False)
    fold_i =0
    for train_id, test_id in kf.split(X):
        fold_i += 1
        X_train, X_test = X[train_id], X[test_id]
        Y_train, Y_test = Y[train_id], Y[test_id]

        model_1 = linear_model.Ridge(alpha)
        model_1.fit(X_train,Y_train)

        Y_test_pred  = model_1.predict(X_test)
        Y_train_pred = model_1.predict(X_train)

        test_rmse.append ( np.sqrt( metrics.mean_squared_error(Y_test , Y_test_pred )))
        train_rmse.append( np.sqrt( metrics.mean_squared_error(Y_train, Y_train_pred)))

    avg_test_RMSE   = sum(test_rmse)/10
    avg_train_RMSE  = sum(train_rmse)/10

    return avg_test_RMSE,avg_train_RMSE



l1_ratio =0

RMSE_ridge ={}

alpha = 0

for wf in np.arange(5):

    RMSE_wf = np.zeros((11,3))
    i=0
    for order in range(1, 12):
        poly = PolynomialFeatures(order)
        X_poly = poly.fit_transform(X_list[wf])
        best_RMSE_test_i,best_RMSE_train_i = kfold_ridg(X_poly,Y_list[wf],alpha)
            
        RMSE_wf[i,0] = order
        RMSE_wf[i,1] = best_RMSE_test_i
        RMSE_wf[i,2] = best_RMSE_train_i
        i += 1        
        
    RMSE_column = RMSE_wf[:,1]
    RMSE_ind    = np.argmin(RMSE_column)
    best_order  = RMSE_wf[RMSE_ind,0]
    best_test_err  = RMSE_wf[RMSE_ind,1]
    avg_train_err  = RMSE_wf[RMSE_ind,2]

    RMSE_ridge['work_flow_%d'%wf] = RMSE_wf
    
    print("\nRMSE_ridge generated wf = ", wf)
    print("Best test error = ",best_test_err,
          "\nAvg train error = ",avg_train_err,
          "\norder = ",best_order,
          )
        
    
    

    

for wf in np.arange(5):

    RMSE_wf = RMSE_ridge['work_flow_%d'%wf] 
    plt.figure()
    plt.title('work_flow_%d'%wf)
    plt.plot(RMSE_wf[:,0],RMSE_wf[:,1] , color = 'green')
    plt.plot(RMSE_wf[:,0],RMSE_wf[:,2] , color = 'blue')
    plt.xlabel("Order")
    plt.ylabel("RMSE")
    
    
    
    
    
    
# More test of work_flow_1 higher order
    
    
l1_ratio =0

RMSE_ridge ={}

alpha = 0

wf =1
    
    
RMSE_wf = np.zeros((14,3))
i=0
for order in range(1, 15):
    poly = PolynomialFeatures(order)
    X_poly = poly.fit_transform(X_list[wf])
    best_RMSE_test_i,best_RMSE_train_i = kfold_ridg(X_poly,Y_list[wf],alpha)
        
    RMSE_wf[i,0] = order
    RMSE_wf[i,1] = best_RMSE_test_i
    RMSE_wf[i,2] = best_RMSE_train_i
    i += 1        
    
    print("order = ",order,"done")
    
RMSE_column = RMSE_wf[:,1]
RMSE_ind    = np.argmin(RMSE_column)
best_order  = RMSE_wf[RMSE_ind,0]
best_test_err  = RMSE_wf[RMSE_ind,1]
avg_train_err  = RMSE_wf[RMSE_ind,2]

RMSE_ridge['work_flow_%d'%wf] = RMSE_wf

print("\nRMSE_ridge generated wf = ", wf)
print("Best test error = ",best_test_err,
      "\nAvg train error = ",avg_train_err,
      "\norder = ",best_order,
      )

    
    
RMSE_wf = RMSE_ridge['work_flow_%d'%wf] 
plt.figure()
plt.title('work_flow_%d'%wf)
plt.plot(RMSE_wf[:,0],RMSE_wf[:,1] , color = 'green')
plt.plot(RMSE_wf[:,0],RMSE_wf[:,2] , color = 'blue')
plt.xlabel("Order")
plt.ylabel("RMSE")
    



print("\n\n")
""" ====== plotting predicted vs true value ====== """
print("====== plot pred vs true output ======")



alpha =0
order_all = np.array([7,9,6,5,6])
    
for wf in np.arange(5):

    
    poly   = PolynomialFeatures( order_all[wf] )
    X_poly = poly.fit_transform(  X_list[wf]   )
    model_1 = linear_model.Ridge(alpha)
    model_1.fit(X_poly,Y_list[wf])
    Y_pred_1  = model_1.predict(X_poly)

    plt.figure()
    yy = Y_pred_1
    
    title_string = 'work_flow_%d'%wf + ', order = %d'%order_all[wf]
    
    plt.title(title_string)
    xx = Y_list[wf]
    plt.scatter(xx, yy, s = 1,  alpha=0.01)
    plt.xlabel('true value')
    plt.ylabel('fitted value')
    xi = range(0,2)
    yi = [i for i in xi]
    plt.axis([-0.03,1.03,-0.03,1.03])
    plt.plot(xi,yi,color='red')
    
    plt.figure()
    xx = Y_pred_1
    yy = np.abs(xx-Y_list[wf])
    plt.scatter(xx, yy, s=1, alpha=0.01)
    plt.axis([-0.03,1.03,-0.03,1.03])
    plt.xlabel('fitted value')
    plt.ylabel('residuals')


print("completed")

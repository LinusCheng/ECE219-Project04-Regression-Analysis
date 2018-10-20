import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
#from sklearn import preprocessing
from sklearn.feature_selection import f_regression, mutual_info_regression
import matplotlib.pyplot as plt




""" ====== Change to scalar ====== """

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


content_all_np = content.values
X = content_all_np[: , 0:5]
Y = content_all_np[: ,  5 ]


del WorkFlow_list,File_list,content_all_np, num_WorkFlow,num_File,i
# delete unused variables

print("\n")
""" ====== <i> Linear Regression ====== """
print("====== <i> Linear Regression ======")




def kfold_lin_reg(X,Y):

    test_rmse  = []
    train_rmse = []

    kf = KFold(n_splits=10, random_state=None, shuffle=False)
    fold_i =0
    for train_id, test_id in kf.split(X):
        fold_i += 1
        X_train, X_test = X[train_id], X[test_id]
        Y_train, Y_test = Y[train_id], Y[test_id]

        model_1 = linear_model.LinearRegression()
        model_1.fit(X_train,Y_train)

        Y_test_pred  = model_1.predict(X_test)
        Y_train_pred = model_1.predict(X_train)

        test_rmse.append ( np.sqrt( metrics.mean_squared_error(Y_test , Y_test_pred )))
        train_rmse.append( np.sqrt( metrics.mean_squared_error(Y_train, Y_train_pred)))

        print("Test_RMSE  fold:",fold_i," = ",test_rmse[-1] )
        print("Train_RMSE fold:",fold_i," = ",train_rmse[-1])
        
    all_test_RMSE  = sum(test_rmse)/10
    all_train_RMSE = sum(train_rmse)/10

    print("\n")
    print("Overall Test_RMSE  = ", all_test_RMSE  )
    print("Overall Train_RMSE = ", all_train_RMSE )
    
    return test_rmse, train_rmse, all_test_RMSE


test_rmse_q1,train_rmse_q1, all_test_RMSE_q1 = kfold_lin_reg(X,Y)





print("\n\n")
""" ====== <ii> Data Preprocessing ====== """
print("====== <ii> Data Preprocessing ======")


scaler = StandardScaler()
#scaler.fit_transform(X)


test_rmse_q2,train_rmse_q2,all_test_RMSE_q2 = kfold_lin_reg(scaler.fit_transform(X),Y)




print("\n\n")
""" ====== <iii> Feature Selection ====== """
print("====== <iii> Feature Selection ======")


f_test, _ = f_regression(X, Y)
f_test /= np.max(f_test)
print("f_test")
print(f_test)

# best 123

m_test = mutual_info_regression(X, Y)
m_test /= np.max(m_test)
print("m_test")
print(m_test)

# best 234


print("\n\n")


test_rmse_q3_fs,train_rmse_q3_fs,all_test_RMSE_q3_fs = kfold_lin_reg(X[:,1:4],Y)
test_rmse_q3_mi,train_rmse_q3_mi,all_test_RMSE_q3_mi = kfold_lin_reg(X[:,2:5],Y)




print("\n\n")
""" ====== plotting predicted vs true value ====== """
print("====== plot pred vs true output ======")


model_1 = linear_model.LinearRegression()
model_1.fit(X,Y)
Y_pred_1  = model_1.predict(X)

scaler = StandardScaler()
model_2 = linear_model.LinearRegression()
model_2.fit(scaler.fit_transform(X),Y)
Y_pred_2  = model_2.predict(scaler.fit_transform(X))

model_3 = linear_model.LinearRegression()
model_3.fit(X[:,1:4],Y)
Y_pred_3  = model_3.predict(X[:,1:4])

model_4 = linear_model.LinearRegression()
model_4.fit(X[:,2:5],Y)
Y_pred_4  = model_4.predict(X[:,2:5])



print("===1===")

plt.figure()
yy = Y_pred_1
plt.title("i linear regression")
xx = Y
plt.scatter(xx, yy, s = 1,  alpha=0.01)
plt.xlabel('true value')
plt.ylabel('fitted value')
xi = range(0,2)
yi = [i for i in xi]
plt.axis([-0.03,1.03,-0.03,1.03])
plt.plot(xi,yi,color='red')

plt.figure()
xx = Y_pred_1
yy = np.abs(xx-Y)
plt.scatter(xx, yy, s=1, alpha=0.01)
plt.axis([-0.03,1.03,-0.03,1.03])
plt.xlabel('fitted value')
plt.ylabel('residuals')

##########
print("===2===")

plt.figure()
yy = Y_pred_2
plt.title("ii standardized linear regression")
xx = Y
plt.scatter(xx, yy, s = 1,  alpha=0.01)
plt.xlabel('true value')
plt.ylabel('fitted value')
xi = range(0,2)
yi = [i for i in xi]
plt.axis([-0.03,1.03,-0.03,1.03])
plt.plot(xi,yi,color='red')

plt.figure()
xx = Y_pred_2
yy = np.abs(xx-Y)
plt.scatter(xx, yy, s=1, alpha=0.01)
plt.axis([-0.03,1.03,-0.03,1.03])
plt.xlabel('fitted value')
plt.ylabel('residuals')

##########
print("===3===")

plt.figure()
yy = Y_pred_3
plt.title("iii f_regression")
xx = Y
plt.scatter(xx, yy, s = 1,  alpha=0.01)
plt.xlabel('true value')
plt.ylabel('fitted value')
xi = range(0,2)
yi = [i for i in xi]
plt.axis([-0.03,1.03,-0.03,1.03])
plt.plot(xi,yi,color='red')

plt.figure()
xx = Y_pred_3
yy = np.abs(xx-Y)
plt.scatter(xx, yy, s=1, alpha=0.01)
plt.axis([-0.03,1.03,-0.03,1.03])
plt.xlabel('fitted value')
plt.ylabel('residuals')


##########
print("===4===")

plt.figure()
yy = Y_pred_4
plt.title("iii mutual_info_regression")
xx = Y
plt.scatter(xx, yy, s = 1,  alpha=0.01)
plt.xlabel('true value')
plt.ylabel('fitted value')
xi = range(0,2)
yi = [i for i in xi]
plt.axis([-0.03,1.03,-0.03,1.03])
plt.plot(xi,yi,color='red')

plt.figure()
xx = Y_pred_4
yy = np.abs(xx-Y)
plt.scatter(xx, yy, s=1, alpha=0.01)
plt.axis([-0.03,1.03,-0.03,1.03])
plt.xlabel('fitted value')
plt.ylabel('residuals')



print("completed")




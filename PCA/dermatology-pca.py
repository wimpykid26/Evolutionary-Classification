import pandas as pd
import plotly.plotly as py
from plotly.graph_objs import *
import plotly
import numpy as np
plotly.tools.set_credentials_file(username='iwayankit', api_key='9syhwIKBYVUPY7uX20I9')
import plotly.tools as tls

df = pd.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data',
    header=None,
    sep=',',
    error_bad_lines = False)

df.colums=['erythema','scaling','definite_borders','itching','koebner_phenomenon','polygonal_papules','follicular_papules','oral_mucosal_involvement','knee_and_elbow_involvement','scalp_involvement','family_history','melanin_incontinence','eosinophils_in_the_infiltrate', 'd', 'e','f','g', 'h', 'i','j','k', 'l', 'm','n','o', 'p', 'q','r','s', 't','u','v','w', 'Age', 'class']
df.dropna(how="all", inplace=True) # drops the empty line at file-end
#cols =  df.columns.tolist()
#cols = cols[-1:] + cols[:-1]
#df = df[cols]
df.replace('?',int('-9999'),inplace=True)

# split data table into data X and class labels y

X = df.ix[:,0:34].values
y = df.ix[:,34].values

from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)

# plotting histograms

traces = []

legend = {0:False, 1:False, 2:False, 3:False ,4:False ,5:False ,6:True }

colors = {1 : 'rgb(31, 119, 180)',
          2 : 'rgb(255, 127, 14)',
          3 : 'rgb(44, 160, 44)',
          4 : 'rgb(31, 221, 180)',
          5 : 'rgb(255, 160, 14)',
          6 : 'rgb(44, 127, 44)'}

for col in range(7):
    for key in colors:
        traces.append(Histogram(x=X[y==key, col],
                        opacity=0.75,
                        xaxis='x%s' %(col+1),
                        marker=Marker(color=colors[key]),
                        name=key,
                        showlegend=legend[col]))

data = Data(traces)
#print data

mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)
print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))

cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)


cor_mat1 = np.corrcoef(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cor_mat1)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

#cor_mat2 = np.corrcoef(X.T)

#eig_vals, eig_vecs = np.linalg.eig(cor_mat2)

#print('Eigenvectors \n%s' %eig_vecs)
#print('\nEigenvalues \n%s' %eig_vals)

u,s,v = np.linalg.svd(X_std.T)
u
for ev in eig_vecs:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
print('Everything ok!')

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

trace1 = Bar(
        x=['PC %s' %i for i in range(1,35)],
        y=var_exp,
        showlegend=False)

trace2 = Scatter(
        x=['PC %s' %i for i in range(1,35)],
        y=cum_var_exp,
        name='cumulative explained variance')

data = Data([trace1, trace2])

layout=Layout(
        yaxis=YAxis(title='Explained variance in percent'),
        title='Explained variance by different principal components')

fig = Figure(data=data, layout=layout)
py.plot(fig)


matrix_w = np.hstack((eig_pairs[0][1].reshape(35,1),
                      eig_pairs[1][1].reshape(35,1),
                      eig_pairs[2][1].reshape(35,1),
                      eig_pairs[3][1].reshape(35,1),
                      eig_pairs[4][1].reshape(35,1),
                      eig_pairs[5][1].reshape(35,1)))

print('Matrix W:\n', matrix_w)

Y = X_std.dot(matrix_w)

traces = []

for name in ('psoriasis', 'seboreic-dermatitis', 'lichen-planus', 'pityriasis-rosea', 'cronic-dermatitis', 'pityriasis-rubra-pilaris'):

    trace = Scatter(
        x=Y[y==name,0],
        y=Y[y==name,1],
        mode='markers',
        name=name,
        marker=Marker(
            size=12,
            line=Line(
                color='rgba(217, 217, 217, 0.14)',
                width=0.5),
            opacity=0.8))
    traces.append(trace)


data = Data(traces)
layout = Layout(showlegend=True,
                scene=Scene(xaxis=XAxis(title='PC1'),
                yaxis=YAxis(title='PC2'),))

fig = Figure(data=data, layout=layout)
py.plot(fig)


layout = Layout(showlegend = True,
                barmode='overlay',
                xaxis=XAxis(domain=[0, .03], title='erythema (cm)'),
                xaxis2=XAxis(domain=[.03, .06], title='scaling (cm)'),
                xaxis3=XAxis(domain=[.09, .12], title='definite_borders (cm)'),
                xaxis4=XAxis(domain=[.12, .15], title='itching (cm)'),
                xaxis5=XAxis(domain=[.18, .21], title='koebner_phenomenon (cm)'),
                xaxis6=XAxis(domain=[.21, .24], title='polygonal_papules (cm)'),
                xaxis7=XAxis(domain=[.24, .27], title='follicular_papules (cm)'),
                xaxis8=XAxis(domain=[.3, .33], title='oral_mucosal_involvement (cm)'),
                xaxis9=XAxis(domain=[.33, .36], title='knee_and_elbow_involvement (cm)'),
                xaxis10=XAxis(domain=[.36, .39], title='scalp_involvement (cm)'),
                xaxis11=XAxis(domain=[.42, .45], title='family_history (cm)'),
                xaxis12=XAxis(domain=[.45, .48], title='melanin_incontinence (cm)'),
                xaxis13=XAxis(domain=[.48, .51], title='eosinophils_in_the_infiltrate (cm)'),
                xaxis14=XAxis(domain=[.51, .54], title='d (cm)'),
                xaxis15=XAxis(domain=[.54, .57], title='e (cm)'),
                xaxis16=XAxis(domain=[.57, .6], title='f (cm)'),
                xaxis17=XAxis(domain=[.6, .63], title='g (cm)'),
                xaxis18=XAxis(domain=[.63, .66], title='h (cm)'),
                xaxis19=XAxis(domain=[.66, .69], title='i (cm)'),
                xaxis20=XAxis(domain=[.69, .72], title='j (cm)'),
                xaxis21=XAxis(domain=[.72, .75], title='k (cm)'),
                xaxis22=XAxis(domain=[.75, .78], title='l (cm)'),
                xaxis23=XAxis(domain=[.78, .81], title='m (cm)'),
                xaxis24=XAxis(domain=[.81, .84], title='n (cm)'),
                xaxis25=XAxis(domain=[.84, .87], title='o (cm)'),
                xaxis26=XAxis(domain=[.87, .90], title='p (cm)'),
                xaxis27=XAxis(domain=[.90, .93], title='q (cm)'),
                xaxis28=XAxis(domain=[.93, .96], title='r (cm)'),
                xaxis29=XAxis(domain=[.96, .99], title='s (cm)'),
                xaxis30=XAxis(domain=[.99, 1.02], title='t (cm)'),
                xaxis31=XAxis(domain=[1.02, 1.05], title='u (cm)'),
                xaxis32=XAxis(domain=[1.05, 1.08], title='v (cm)'),
                xaxis33=XAxis(domain=[1.08, 1.11], title='w (cm)'),
                xaxis34=XAxis(domain=[1.11, 1.14], title='Age (cm)'),
                yaxis=YAxis(title='count'),
                title='Distribution of the dermatology attributes')

fig = Figure(data=data, layout=layout)
#py.plot(fig)



df.tail()


from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import xlrd
from sklearn.model_selection import train_test_split

data = xlrd.open_workbook('4C_Dataset.xlsx')
table_data = data.sheets()[0]
data_nrows = table_data.nrows
data_ncols = table_data.ncols

data_datamatrix=np.zeros((data_nrows,data_ncols))

for x in range(data_ncols):
    data_cols =table_data.col_values(x)    
 
    data_cols1=np.matrix(data_cols)

    data_datamatrix[:,x]=data_cols1
        
species_data=np.zeros((data_nrows,1))
species_data=data_datamatrix[:,0]-1

y_species_data=tf.one_hot(species_data,18,on_value=1,off_value=None,axis=1)

with tf.Session()as sess:
    y_data = y_species_data.eval()

X_data=np.zeros((data_nrows,75))
X_data=data_datamatrix[:,1:76]

x_train, x_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=42)

train_nrows = len(x_train)

learning_rate = 0.00000001
num_steps =8000
batch_size =64
display_step = 8000
examples_to_show = 10

num_input = 75

num_hidden_1 =1024

num_code=1024

num_hidden_2 =1024

num_hidden_3 =1024

num_output = 18

beta = 0.00015

train_loss=np.zeros((num_steps//10,1))
test_loss=np.zeros((num_steps//10,1))

def weight_variable(shape,name):
    initial=tf.truncated_normal(shape,stddev=0.015)
    return tf.Variable(initial,name=name)

def bias_variable(shape,name):
    initial=tf.constant(0.3,shape=shape)
    return tf.Variable(initial,name=name)

keep_prob = tf.placeholder(tf.float32)
keep_prob = 1

with tf.name_scope('input'):
    x=tf.placeholder(tf.float32,[None,num_input],name='x_input')
    y=tf.placeholder(tf.float32,[None,num_output],name='y_input')
    
with tf.name_scope('hidden_1'):
    w1=weight_variable([num_input,num_hidden_1],name='w1')
    b1=bias_variable([num_hidden_1],name='b1')
    with tf.name_scope('node_1'):
        x = tf.nn.dropout(x, keep_prob)
        node_1=tf.matmul(x,w1)+b1
    with tf.name_scope('relu'):
        h_1=tf.nn.relu(node_1)
        
with tf.name_scope('shortcut1'): 
    w11=weight_variable([num_hidden_1,num_hidden_2],name='w11')
    b11=bias_variable([num_hidden_2],name='b11')
    with tf.name_scope('node_11'):
        h_11 = tf.nn.dropout(h_1, keep_prob)
        node_11=tf.matmul(h_11,w11)+b11
    with tf.name_scope('relu'):
        h_11=tf.nn.relu(node_11)


with tf.name_scope('shortcut2'): 
    w12=weight_variable([num_hidden_1,num_output],name='w12')
    b12=bias_variable([num_output],name='b12')
    with tf.name_scope('node_12'):
        h_12 = tf.nn.dropout(h_1, keep_prob) 
        node_12=tf.matmul(h_12,w12)+b12
    with tf.name_scope('relu'):
        h_12=tf.nn.relu(node_12)


with tf.name_scope('encode'):
    w2=weight_variable([num_hidden_1,num_code],name='w2')
    b2=bias_variable([num_code],name='b2')
    with tf.name_scope('sum_encode'):
        h_1 = tf.nn.dropout(h_1, keep_prob)
        sum_encode=tf.matmul(h_1,w2)+b2
    with tf.name_scope('relu'):
        h_encode=tf.nn.relu(sum_encode)
        
with tf.name_scope('shortcut3'): 
    w21=weight_variable([num_code,num_output],name='w21') 
    b21=bias_variable([num_output],name='b21')
    with tf.name_scope('code_1'):
        h_encode2 = tf.nn.dropout(h_encode, keep_prob)
        code_1=tf.matmul(h_encode2,w21)+b21
    with tf.name_scope('relu'):
        h_encode2=tf.nn.relu(code_1)


with tf.name_scope('decode'):
    w3=weight_variable([num_code,num_hidden_2],name='w3')
    b3=bias_variable([num_hidden_2],name='b3')
    with tf.name_scope('sum_decode'):
        h_encode = tf.nn.dropout(h_encode, keep_prob)
        
        sum_decode=tf.matmul(h_encode,w3)+h_11
    with tf.name_scope('relu'):
        h_decode=tf.nn.relu(sum_decode)

with tf.name_scope('shortcut4'): 
    w34=weight_variable([num_hidden_2,num_output],name='w34')
    b34=bias_variable([num_output],name='b34')
    with tf.name_scope('code_2'):
        h_decode1= tf.nn.dropout(h_decode, keep_prob)
        code_2=tf.matmul(h_decode1,w34)+b34
    with tf.name_scope('relu'):
        h_decode1=tf.nn.relu(code_2)

with tf.name_scope('hidden_2'): 
    
    w4=weight_variable([num_hidden_2,num_hidden_3],name='w4')
    b4=bias_variable([num_hidden_3],name='b4')
    with tf.name_scope('node_1'):
        h_decode = tf.nn.dropout(h_decode, keep_prob)
        node_1=tf.matmul(h_decode,w4)+b4
    with tf.name_scope('relu'):
        h_2=tf.nn.relu(node_1)
        
with tf.name_scope('hidden_3'): 
    
    w5=weight_variable([num_hidden_3,num_output],name='w5')
    b5=bias_variable([num_output],name='b5')
    with tf.name_scope('node_1'):
        h_2 = tf.nn.dropout(h_2, keep_prob)
        node_1=tf.matmul(h_2,w5)+b5+h_encode2+h_12+h_decode1
    with tf.name_scope('relu'):
        h_3=tf.nn.relu(node_1)
    with tf.name_scope('prediction'):
        prediction=tf.nn.softmax(h_3)


with tf.name_scope('loss_mean_square'):
    
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction),name='cross_entropy')
    regularizers = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3) +tf.nn.l2_loss(w4)+tf.nn.l2_loss(w5) ##add regularizization
    cross_entropy = tf.reduce_mean(loss + beta * regularizers)
    tf.summary.scalar('cross',cross_entropy)
with tf.name_scope('train'):

    train_step=tf.train.AdamOptimizer(2e-5).minimize(cross_entropy)

   
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction= tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
    with tf.name_scope('accuracy'):
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        tf.summary.scalar('accuracy',accuracy)
merged=tf.summary.merge_all()

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    train_writer=tf.summary.FileWriter('logs/train',sess.graph)
    test_writer=tf.summary.FileWriter('logs/test',sess.graph)

    batch_count=int(train_nrows/batch_size)
    reminder=train_nrows%batch_size
    for i in range(num_steps):
        learning_rate = 0.000000003
        if num_steps == 1000:
            learning_rate = 0.000000003
        elif num_steps == 3000:
            learnin_rate = 0.000000002
        elif num_steps == 5000:
            learnin_rate = 0.000000001

        for n in range(batch_count):
            
            train_step.run(feed_dict={x: x_train[n*batch_size:(n+1)*batch_size], y: y_train[n*batch_size:(n+1)*batch_size]})  

        if reminder>0:
            start_index = batch_count * batch_size;  
            train_step.run(feed_dict={x: x_train[start_index:train_nrows-1], y: y_train[start_index:train_nrows-1]})  
        
        iterate_accuracy = 0 
        if i%10==0:
            train_loss[i//10,0]=sess.run(accuracy,feed_dict={x:x_train,y:y_train})
            test_loss[i//10,0]=sess.run(accuracy,feed_dict={x:x_test,y:y_test})
            print('Iter'+str(i)+', Testing Accuracy= '+str(test_loss[i//10,0])+',Training Accuracy=' +str(train_loss[i//10,0]))
 
    x_index = np.linspace(0, num_steps, num_steps//examples_to_show)
    
    font1 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 32,
    }
    figsize = 8,8
    figure, ax = plt.subplots(figsize=figsize)
    
    A,=plt.plot(x_index, train_loss, color="red",label='train_accuracy',linewidth=2.0,ms=10)
    B,=plt.plot(x_index, test_loss, color="blue",label='test_accuracy',linewidth=2.0,ms=10)
    plt.legend(handles=[A,B],prop=font1)
    plt.xlabel("Iterations", font1)
    plt.ylabel("Accuracy (Species)", font1)
    
    plt.tick_params(labelsize=23)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.show()
    
    
   
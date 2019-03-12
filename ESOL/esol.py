import os, sys
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import time
from rdkit.Chem import AllChem
from rdkit import Chem


# esol regression model ========================
start_time = time.time()

### run this code once at first
'''trfile = open('esol.csv', 'r')
line = trfile.readline()
dataX = []
dataY = []
for line in trfile:
    line = line.rstrip().split(',')
    smiles = str(line[8])
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    fp = np.array(fp)
    dataX.append(fp)
    val = float(line[7])
    dataY.append(val)
dataX = np.array(dataX)
print(dataX.shape)
print(dataX[0])
np.save('esol_fp', dataX)
dataY = np.array(dataY)
np.save('esol_Y', dataY)
sys.exit()
'''

dataX = np.load('esol_fp.npy')
dataY = np.load('esol_Y.npy')
print(dataX.shape, dataY.shape)

batch_size = 32
Max_len = 100
bit_size = 1024
embedding_size = 100
emb = tf.Variable(tf.random_uniform([bit_size+1, embedding_size], -1, 1), dtype=tf.float32)
pads = tf.constant([[1,0], [0,0]])
embeddings = tf.pad(emb, pads)

data_x = []
data_y = []
for i in range(len(dataX)):
    fp = [0] * Max_len
    n_ones = 0
    for j in range(bit_size):
        if dataX[i][j] == 1:
            fp[n_ones] = j+1
            n_ones += 1
    data_x.append(fp)
    data_y.append([dataY[i]])
data_x = np.array(data_x, dtype=np.int32)
data_y = np.array(data_y, dtype=np.float32)

train_x_concat, test_x_concat, train_y_concat, test_y_concat, valid_x_concat, valid_y_concat = [], [], [], [], [], []

train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.1)
train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.1111)
print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)
print(valid_x.shape, valid_y.shape)

train_size = len(train_x)
test_size = len(test_x)
valid_size = len(valid_x)


##################################################################
# ===================== model construction =======================
##################################################################

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev = 0.01))
def bias_variable(shape):
    return tf.Variable(tf.constant(0.01, shape=shape))

p_keep_conv = tf.placeholder(dtype=tf.float32)

class model():
    def __init__(self, embedding_size, p_keep_conv, Max_len):
         self.Max_len = Max_len
         self.nhid  = 2048
         self.kernel_size = [5] 
         self.w2 = init_weights([self.kernel_size[0], embedding_size, 1, self.nhid]) # 64
         self.w_o = init_weights([self.nhid, 1])
         
         self.b2 = bias_variable([1, self.nhid])
         self.b_o = bias_variable([1])
         self.p_keep_conv = p_keep_conv
         
    
    def conv_model(self, X): 
        l2 = tf.nn.relu(tf.nn.conv2d(X, self.w2, strides=[1, 1, 1, 1], padding='VALID') + self.b2)
        l2 = tf.squeeze(l2, [2])
        l2 = tf.nn.pool(l2, window_shape=[self.Max_len-self.kernel_size[0]+1], strides = [3], pooling_type='MAX', padding='VALID') # stride =3 
        l2 = tf.nn.dropout(l2, self.p_keep_conv)
          
        lout = tf.reshape(l2, [-1, self.w_o.get_shape().as_list()[0]])
        return lout

# ============================================================= #
X = tf.placeholder(tf.int32, [None, Max_len])
Y = tf.placeholder(tf.float32, [None, 1])
X_em = tf.nn.embedding_lookup(embeddings, X)
X_em = tf.reshape(X_em, [-1, Max_len, embedding_size, 1])

model = model(embedding_size, p_keep_conv, Max_len)
py_x = model.conv_model(X_em)

# ============================================================= #
temp_hid = 2048
w1 = init_weights([temp_hid, 1])
b1 = bias_variable([1])

py_x1 = tf.matmul(py_x, w1) + b1
cost1 = tf.losses.mean_squared_error(labels=Y, predictions=py_x1)

lr = 5e-4
train_op1 = tf.train.AdamOptimizer(learning_rate = lr).minimize(cost1)
prediction_error = tf.sqrt(cost1)


##################################################################
# ==================== training part =============================
##################################################################
SAVER_DIR = "model_esol"
saver = tf.train.Saver()
ckpt_path = os.path.join(SAVER_DIR, "model_esol")
ckpt = tf.train.get_checkpoint_state(SAVER_DIR)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    best_rmse = 10 
    best_idx = 0
    for i in range(100):
        training_batch = zip(range(0, len(train_x), batch_size),
                             range(batch_size, len(train_x)+1, batch_size))
   #for start, end in tqdm.tqdm(training_batch):
        for start, end in training_batch:
            sess.run(train_op1, feed_dict={X: train_x[start:end], Y: train_y[start:end], p_keep_conv: 0.5})

        merr_valid = sess.run(prediction_error, feed_dict={X: valid_x, Y: valid_y, p_keep_conv: 1.0})

   # print validation loss
        if i % 10 == 0:
            print(i, merr_valid)

        if best_rmse > merr_valid:
            best_idx = i
            best_rmse = merr_valid
            save_path = saver.save(sess, ckpt_path, global_step = best_idx)
            print('model saved!')

print('best RMSE esol: ')
print(best_rmse)
print("=== %s seconds ===" % (time.time() - start_time))
print('best idx: '+str(best_idx))

####################################################################
#=========================== test part ============================#
####################################################################
saver = tf.train.Saver()
ckpt_path = os.path.join(SAVER_DIR, "model_esol")
ckpt = tf.train.get_checkpoint_state(SAVER_DIR)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver.restore(sess, ckpt.model_checkpoint_path)
    print("model loaded successfully!")
    test_rmse = sess.run(prediction_error, feed_dict={X: test_x, Y: test_y, p_keep_conv: 1})
    print("RMSE of test set:")
    print(test_rmse)
                                       

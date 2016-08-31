import os
import cv2
import tensorflow as tf

IMG_DIR = '../data'
MODEL_PATH = 'models/classify_image_graph_def.pb'

QUERY_IMG = 22
CANDIDATES = 5

with tf.gfile.FastGFile(MODEL_PATH, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    pool3 = sess.graph.get_tensor_by_name('pool_3:0')
    features = []
    directories = os.listdir(IMG_DIR)
    for d in directories:
        if os.path.isdir(IMG_DIR+"/"+d) == False:
            directories.remove(d)
    for d in directories:
        files = os.listdir(IMG_DIR+"/"+d)
        for f in files:
            print f
            image_data = tf.gfile.FastGFile('{0}/{1}/{2}'.format(IMG_DIR, d, f), 'rb').read()
            pool3_features = sess.run(pool3,{'DecodeJpeg/contents:0': image_data})
            features.append(np.squeeze(pool3_features))

query_feat = features[QUERY_IMG]
sims = [(k, round(1 - spatial.distance.cosine(query_feat, v), 3)) for k,v in enumerate(features)]
print(sorted(sims, key=operator.itemgetter(1), reverse=True)[:CANDIDATES + 1])

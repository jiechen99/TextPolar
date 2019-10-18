import cv2
import time
import math
import os
import numpy as np
import tensorflow as tf
import scipy.misc
import locality_aware_nms as nms_locality
import lanms

tf.app.flags.DEFINE_string('test_data_path', '/tmp/ch4_test_images/images/', '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/east_icdar2015_resnet_v1_50_rbox/', '')
tf.app.flags.DEFINE_string('output_dir', '/tmp/ch4_test_images/images/', '')
tf.app.flags.DEFINE_bool('no_write_images', False, 'do not write images')

import model

FLAGS = tf.app.flags.FLAGS

def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape
    resize_w = w
    resize_h = h

    '''
    # limit the max side
    
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    '''
    ratio = math.sqrt((512.0 * 512.0) / (resize_h * resize_w))
    #ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)
    
    #resize_h = 512.
    #resize_w = 512.
    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)

def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list


    try:
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise

    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        f_center, f_skeleton = model.model(input_images, is_training=False)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            #ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            #model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            model_path = FLAGS.checkpoint_path
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            im_fn_list = get_images()
            for im_fn in im_fn_list:
                print im_fn
                im = cv2.imread(im_fn)[:, :, ::-1]
                im_resized, (ratio_h, ratio_w) = resize_image(im)

                timer = {'net': 0, 'restore': 0, 'nms': 0}
                start = time.time()
                center, skeleton = sess.run([f_center, f_skeleton],
                                                      feed_dict={input_images: [im_resized]})
                center_preds = np.squeeze(center)
                skeleton_preds = np.squeeze(skeleton)
		center_preds[center_preds<0.8] = 0
		skeleton_preds[skeleton_preds<0.8] = 0
		#print 'skeleton_preds: {}'.format(skeleton_preds)
                cv2.imwrite(os.path.join(FLAGS.output_dir, os.path.basename(im_fn))[:-4]+'_skeleton.jpg', skeleton_preds*255)
                cv2.imwrite(os.path.join(FLAGS.output_dir, os.path.basename(im_fn))[:-4]+'_center.jpg', center_preds*255)
                '''
                timer['net'] = time.time() - start
                score_f = np.squeeze(score_f)
                print 'max of score_f: {}  min :{}'.format(score_f.max(), score_f.min())
                f_score_img = np.expand_dims(score_f, axis=2)
                h, w, _ = f_score_img.shape
                blank = np.zeros((int(h), int(w), 1))
                print 'f_score_img.shape : {}'.format(f_score_img.shape)
                print 'blank.shape: {}'.format(blank.shape)
                output_f_score = np.concatenate((f_score_img, blank, blank), axis=2)
                #scipy.misc.imsave(os.path.join(FLAGS.output_dir, os.path.basename(im_fn))[:-4]+'_f.jpg', output_f_score)

                score_b = np.squeeze(score_b)
                #b_mean = np.mean(score_b)
                #score_b[score_b<b_mean]=0
                #score_b[score_b>b_mean]=1
                b_score_img = np.expand_dims(score_b, axis=2)
                #b_score_img = cv2.resize(b_score_img, (int(w*4/ratio_w), int(h*4/ratio_h)))

                #b_score_img = np.expand_dims(b_score_img, axis=2)

                output_b_score = np.concatenate((blank, b_score_img, blank), axis=2)
               # scipy.misc.imsave(os.path.join(FLAGS.output_dir, os.path.basename(im_fn))[:-4]+'_b.jpg', output_b_score)

                final = output_f_score + output_b_score
                final = cv2.resize(final, (int(w*4/ratio_w), int(h*4/ratio_h)))
                mean_0 = np.mean(final[::, ::, 0])
                mean_1 = np.mean(final[::, ::, 1])
                final[::, ::, 0][final[::, ::, 0] > mean_0] = 1
                final[::, ::, 0][final[::, ::, 0] < mean_0] = 0
                final[::, ::, 1][final[::, ::, 1] > mean_1] = 1
                final[::, ::, 1][final[::, ::, 1] < mean_1] = 0
                scipy.misc.imsave(os.path.join(FLAGS.output_dir, os.path.basename(im_fn))[:-4]+'_fb.jpg', final)

                final = cv2.imread(os.path.join(FLAGS.output_dir, os.path.basename(im_fn))[:-4]+'_fb.jpg')
                im = cv2.imread(im_fn)
                h, w, c = final.shape
                #mean = [np.mean(final[::,::,0]), np.mean(final[::,::,1]),np.mean(final[::,::,2])]
                for ci in range(c):
                    for hi in range(h):
                        for wi in range(w):
                            if final[hi][wi][ci] > 255/2:
                                im[hi][wi][ci] = final[hi][wi][ci]
                                for kk in range(3):
                                    if im[hi][wi][kk] != final[hi][wi][kk]:
                                        im[hi][wi][kk] = 0
                cv2.imwrite(os.path.join(FLAGS.output_dir, os.path.basename(im_fn))[:-4]+'_fb2.jpg', im)
                '''
if __name__ == '__main__':
    tf.app.run()

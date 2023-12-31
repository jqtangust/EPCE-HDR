import os
import os.path as osp
import numpy as np
import data_io as io
import cv2
import metrics as m

ref_dir = "/home/jiaqitang/Real_Valset_1024/GT"
ref_alignratio_dir = '/home/jiaqitang/Real_Valset_1024/alignratio'
# res_dir = './Visualization/medium'
# res_alignratio_dir = ref_alignratio_dir
res_dir = '/home/jiaqitang/Final_Ours/results/Ours-wild-under/input'
res_alignratio_dir = '/home/jiaqitang/Final_Ours/results/Ours-wild-under/restore'

ref_output_dir = '/home/jiaqitang/Final_Ours/Visualization/tone_mapped_gt'
# res_output_dir = './Validation/tone_mapped_medium'
res_output_dir = '/home/jiaqitang/Final_Ours/Visualization/tone_mapped_results_real_wild_under'

if not osp.exists(ref_output_dir):
    os.mkdir(ref_output_dir)

if not osp.exists(res_output_dir):
    os.mkdir(res_output_dir)

for filename in sorted(os.listdir(res_dir)):
    image_id = int(filename[:3])
    # print(image_id)
    # print(osp.join(ref_dir, "{:04d}_gt.png".format(image_id)))
    # print(osp.join(ref_alignratio_dir,
    #       "{:04d}_alignratio.npy".format(image_id)))
    # hdr_image = io.imread_uint16_png(osp.join(ref_dir, "{:04d}_gt.png".format(
    #     image_id)), osp.join(ref_alignratio_dir, "{:04d}_alignratio.npy".format(image_id)))
    # hdr_linear_image = hdr_image ** 2.24
    # norm_perc = np.percentile(hdr_linear_image, 99)

    # hdr_image = (m.tanh_norm_mu_tonemap(hdr_linear_image,
    #              norm_perc) * 255.).round().astype(np.uint8)
    # cv2.imwrite(osp.join(ref_output_dir, "{:04d}_tone_mapped_gt.png".format(
    #     image_id)),  cv2.cvtColor(hdr_image, cv2.COLOR_RGB2BGR))

    # res_image = cv2.cvtColor(cv2.imread(osp.join(res_dir, "{:04d}_medium.png".format(image_id))), cv2.COLOR_BGR2RGB) / 255.

    res_image = io.imread_uint16_png(osp.join(res_dir, "{:03d}.png".format(
        image_id)), osp.join(res_alignratio_dir, "{:03d}_alignratio.npy".format(image_id)))
    res_linear_image = res_image ** 2.24
    hdr_linear_image = res_image ** 2.24
    norm_perc = np.percentile(hdr_linear_image, 99)
    res_image = (m.tanh_norm_mu_tonemap(res_linear_image,
                 norm_perc) * 255.).round().astype(np.uint8)
    cv2.imwrite(osp.join(res_output_dir, "{:03d}_tone_mapped_result.png".format(
        image_id)),  cv2.cvtColor(res_image, cv2.COLOR_RGB2BGR))

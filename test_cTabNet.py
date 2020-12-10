from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv

# Load model

if __name__ == '__main__':
    config_file = './cascade_mask_rcnn_hrnetv2p_w32_20e.py'
    checkpoint_file = './new_epoch_36.pth'
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    img = './merged.jpg'
    result = inference_detector(model, img)
    print(result)

# # Test a single image
# img = "/content/CascadeTabNet/Demo/demo.png"
#
# # Run Inference
# result = inference_detector(model, img)

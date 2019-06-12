

def test_multi_pose():
    import sys
    CENTERNET_PATH = '/app/code/src/lib/'
    sys.path.insert(0, CENTERNET_PATH)

    from detectors.detector_factory import detector_factory
    from opts import opts

    MODEL_PATH = 'models/multi_pose_dla_3x.pth'
    TASK = 'multi_pose' # or 'ctdet' for human pose estimation
    opt = opts().init('{} --load_model {}'.format(TASK, MODEL_PATH).split(' '))
    detector = detector_factory[opt.task](opt)

    img = 'images/33823288584_1d21cf0a26_k.jpg'
    ret = detector.run(img)['results']

    import pdb; pdb.set_trace()


test_multi_pose()

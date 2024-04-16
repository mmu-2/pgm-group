```bash
python train.py --dataroot ./datasets/smoking --name smoking --model cycle_gan --display_id -1 --use_diffaug --use_specnorm
```


```bash
python test.py --dataroot ./datasets/smoking --name smoking --model cycle_gan
```

```bash
python cas.py --dataroot ./results/smoke-pop/test_latest/images --dataroot2 ./datasets/smoking --output_dir ./cas/smoking/
```
    parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    parser.add_argument('--dataroot2', required=True, help='path to second set of images that we will also evaluate on')
    parser.add_argument('--epochs', type=int, default=100, help="number of epochs (expect around 200 images per epoch)")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default='./cas/smoking/')
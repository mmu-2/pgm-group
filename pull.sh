myArray=(
        "pop-smoke-results" \
        "cigarette-processed-dataset" \
        "popsicle_stick_output_processed" \
        "test-results" \
        );
select opt in ${myArray[@]};
do
  case $opt in
    "pop-smoke-results")
        source ~/miniconda3/etc/profile.d/conda.sh
        conda activate cyclegan
        echo "Pulling $opt..."
        mkdir -p ./pytorch-CycleGAN-and-pix2pix/results
        gdown 1_8UcQSmM5W5QAFmMW-lPjR8NWwcynOCJ -O ./pytorch-CycleGAN-and-pix2pix/temp/download.zip
        unzip -q ./pytorch-CycleGAN-and-pix2pix/temp/download.zip -d \
          ./pytorch-CycleGAN-and-pix2pix/results
        ;;
    "cigarette-processed-dataset")
        source ~/miniconda3/etc/profile.d/conda.sh
        conda activate cyclegan
        echo "Pulling $opt..."
        mkdir -p ./pytorch-CycleGAN-and-pix2pix/results
        gdown 1v46TuirhKAs3WLIj_Z-oRpPKSRL8Jjnj -O ./pytorch-CycleGAN-and-pix2pix/temp/download.zip
        unzip -q ./pytorch-CycleGAN-and-pix2pix/temp/download.zip -d \
          ./pytorch-CycleGAN-and-pix2pix/results/cigarette-processed-dataset
        ;;
    "popsicle_stick_output_processed")
        source ~/miniconda3/etc/profile.d/conda.sh
        conda activate cyclegan
        echo "Pulling $opt..."
        mkdir -p ./pytorch-CycleGAN-and-pix2pix/results
        gdown 1C5ojIGzAKWLGqgit1fHCTh34AcFCXvyk -O ./pytorch-CycleGAN-and-pix2pix/temp/download.zip
        unzip -q ./pytorch-CycleGAN-and-pix2pix/temp/download.zip -d \
          ./pytorch-CycleGAN-and-pix2pix/results/popsicle_stick_output_processed
        ;;
    "test-results")
        source ~/miniconda3/etc/profile.d/conda.sh
        conda activate cyclegan
        echo "Pulling $opt..."
        mkdir -p ./pytorch-CycleGAN-and-pix2pix/results
        gdown 1qDcLvhyso1qyywL82US8-bTQ1OgE2nFT -O ./pytorch-CycleGAN-and-pix2pix/temp/download.zip
        unzip -q ./pytorch-CycleGAN-and-pix2pix/temp/download.zip -d \
          ./pytorch-CycleGAN-and-pix2pix/results/outputs/test-results
        ;;
    *) echo invalid option;;
  esac
  break
done
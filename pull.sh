myArray=("pop-smoke-results");
select opt in $myArray;
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
    "Quit")
        break
        ;;
    *) echo invalid option;;
  esac
  break
done
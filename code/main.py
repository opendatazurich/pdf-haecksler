import logging
logging.disable(logging.INFO)
from functions import *
import warnings
warnings.filterwarnings("ignore")
transformers.utils.logging.set_verbosity_error()

######################
## GLOBAL VARIABLES ##
######################

INPUT_PATH = os.getenv("INPUT_PATH")                    # "../../_data/raw/test/"
OUTPUT_PATH = os.getenv("OUTPUT_PATH")                  # "../../_data/outputs/"
CROP_OFFSET = float(os.getenv("CROP_OFFSET"))           # 0.03 (percentage)
THUMB_SIZE = int(os.getenv("THUMB_SIZE"))               # 190 (pixels)
THRESHOLD = float(os.getenv("THRESHOLD"))               # 0.75
PROCESSING_RES = int(os.getenv("PROCESSING_RES"))       # 300 (dpi)
COMPRESSION_LEVEL = int(os.getenv("COMPRESSION_LEVEL")) # 4
COMPRESSION_WAIT = int(os.getenv("COMPRESSION_WAIT"))   # 300 (seconds)
MODEL = "model/model.ckpt"
MODEL_BASIS = "facebook/detr-resnet-50"

###################
## LOAD ML MODEL ##
###################

print("\nLoading Model...\n")
labels_no = 2
feature_extractor = DetrFeatureExtractor.from_pretrained(MODEL_BASIS)
model = Detr(labels_no, MODEL_BASIS)
model_trained = model.load_from_checkpoint(MODEL,num_labels=labels_no,model_basis=MODEL_BASIS).eval()
BRAIN = {"model":model_trained, "feature_extractor":feature_extractor}
print("\n\n")

##################
## RUN CROPPING ##
##################

res = crop_documents(INPUT_PATH, OUTPUT_PATH, BRAIN, THRESHOLD, THUMB_SIZE,
                     CROP_OFFSET, PROCESSING_RES, COMPRESSION_LEVEL, COMPRESSION_WAIT)
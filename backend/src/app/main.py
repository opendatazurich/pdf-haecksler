from .functions import *
import warnings
warnings.filterwarnings("ignore")
transformers.utils.logging.set_verbosity_error()

######################
## GLOBAL VARIABLES ##
######################

CROP_OFFSET = float(os.getenv("CROP_OFFSET"))           # 0.03 (percentage)
THUMB_SIZE = int(os.getenv("THUMB_SIZE"))               # 190 (pixels)
THRESHOLD = float(os.getenv("THRESHOLD"))               # 0.75 (percentage)
PROCESSING_RES = int(os.getenv("PROCESSING_RES"))       # 300 (dpi)
COMPRESSION_LEVEL = int(os.getenv("COMPRESSION_LEVEL")) # 4 (ghostscript compression category)
COMPRESSION_WAIT = int(os.getenv("COMPRESSION_WAIT"))   # 300 (seconds)
MODEL = "model/model.ckpt"
MODEL_BASIS = "facebook/detr-resnet-50"
OUTPUT_PATH = "data/tmp/"

################
## CATEGORIES ##
################



CATEGORIES = { 0 : { "label":"image"  , "output_fs": {"pdf":{"optimized":{},"original":{}}, "png":{}, "thumbs":{}} },
               1 : { "label":"drawing", "output_fs": {"pdf":{"optimized":{},"original":{}}, "png":{}, "thumbs":{}} }}


###################
## LOAD ML MODEL ##
###################

logging.info("Loading Model...\n")
labels_no = 2
feature_extractor = DetrFeatureExtractor.from_pretrained(MODEL_BASIS)
model = Detr(labels_no, MODEL_BASIS)
model_trained = model.load_from_checkpoint(MODEL,num_labels=labels_no,model_basis=MODEL_BASIS).eval()
BRAIN = {"model":model_trained, "feature_extractor":feature_extractor}
logging.info("\n\n")

####################
## FUNCTION CALLS ##
####################

def process_pdf_stream(stream, fname, user=None):
    logging.info(f"\tUser: {user} - File: {fname}\n")
    user = f"{user}/" if user else ""
    if not os.path.exists(OUTPUT_PATH+user):
        os.mkdir(OUTPUT_PATH+user)
    out = OUTPUT_PATH+user+fname
    crop_document_stream(stream, out, BRAIN, CATEGORIES, THRESHOLD, THUMB_SIZE,
                          CROP_OFFSET, PROCESSING_RES, COMPRESSION_LEVEL, COMPRESSION_WAIT)
    zpath = shutil.make_archive(out,'zip',out)
    shutil.rmtree(OUTPUT_PATH+user) if user!="" else shutil.rmtree(out)
    return zpath

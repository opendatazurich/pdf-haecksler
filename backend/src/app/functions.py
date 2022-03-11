import pytorch_lightning as pl
import os
import os.path
import transformers
from transformers import DetrForObjectDetection, DetrFeatureExtractor
from PIL import Image
import numpy as np
import fitz
import copy
import io
import glob
import matplotlib.pyplot as plt
import re
import argparse
import subprocess
import sys
import shutil
import multiprocessing
import signal
import time
from datetime import datetime
import logging


#############
## Logging ##
#############


dt_string = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(filename=f'src/app/logs/{dt_string}.log', filemode='w', level=logging.INFO)
stderrLogger=logging.StreamHandler()
stderrLogger.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
logging.getLogger().addHandler(stderrLogger)


##########
## Util ##
##########


def timer_print(indentation=""):
    '''
    Decorator for counting execution time of any function

    Parameter:
        indentation (string) : printing indentation (eg. \t\t)
    '''
    def timer(func):
        f_time_report = lambda t1,t2: f"{int(np.divide(t2-t1,60))}m {round(np.mod(t2-t1,60),1)}s"
        def wrapper_function(*args, **kwargs):
            t1 = time.time()
            result = func(*args, **kwargs)
            t2 = time.time()
            logging.info(f"{indentation}ok : {f_time_report(t1,t2)}\n")
            return result
        return wrapper_function
    return timer


class TimeoutException(Exception):
    '''
    Custom Exception for slow functions
    '''
    def __init__(self, *args, **kwargs):
        pass


def timeout_handler(num, stack):
    '''
    Handler for signal module
    '''
    raise TimeoutException


def try_to_compress(wait,function,*args):
    '''
    Break pdf compression if very slow

    Parameters:
        wait (int) : seconds to wait
        function (function) : compression function
    '''
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(wait)
    try:
        function(*args)
    except TimeoutException:
        os.remove(args[1])
        logging.info("\t\t\t! skipping optimization : too slow")
    finally:
        signal.alarm(0)


###############
## Inference ##
###############


class Detr(pl.LightningModule):
    '''
    Loads DETR model

    Parameters:
        num_labels (int) : indicates the output size
        model_basis (string) : the pretrained model name
    '''
    def __init__(self,  num_labels, model_basis):
            super().__init__()
            self.model = DetrForObjectDetection.from_pretrained(model_basis, \
                         num_labels=num_labels, ignore_mismatched_sizes=True)
    def forward(self, pixel_values, pixel_mask):
            outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            return outputs

@timer_print(indentation="\t")
def infer(images, model, feature_extractor, threshold = 0.75):
    '''
    Extracts drawing and image regions from a list of (pdf) images

    Parameters:
        images ([PIL.Image]) : list of PIL.Image objects
        model (Detr) : trained Detr model
        feature_extractor (DetrFeatureExtractor) : Detr feature extractor object from huggingface transformers
        threshold (float) : level of accepted confidence for output predictions

    Returns:
        (list) : of dictionaries with fields ->
            bboxes (list) : bounding boxes,
            labels (list) : labels,
            probs  (list) : probabilities
    '''
    # convert bbox from x_c, y_c, w, h to xmin,ymin,xmax,ymax
    f_bbox = lambda b: np.array([b[:,0] - b[:,2]/2, b[:,1] - b[:,3]/2, b[:,0] + b[:,2]/2, b[:,1] + b[:,3]/2]).T
    _images = [*map(copy.deepcopy,images)]
    pixel_values = feature_extractor(_images,return_tensors="pt")['pixel_values']
    outputs = model(pixel_values=pixel_values, pixel_mask=None)
    probas = outputs.logits.softmax(-1)[:, :, :-1]
    keep = probas.max(-1).values > threshold
    bboxes = [*map(lambda x: f_bbox(x[1][keep[x[0]]].detach().numpy()).tolist() , enumerate(outputs.pred_boxes))]
    labels = [*map(lambda x: x[1][keep[x[0]]].detach().argmax(1).numpy().tolist() , enumerate(probas))]
    probs = [*map(lambda x: x[0][x[1]].detach().numpy().tolist() ,zip(probas,keep))]
    res = [*map(lambda l: {"bboxes": l[0], "labels": l[1], "probs":l[2]}, zip(bboxes,labels,probs))]
    return res


##############
## Plotting ##
##############


def plot_results(image, prediction, lw = 2, fsz = 10, offset=None):
    '''
    Plots image with predicted bounding boxes

    Parameters:
        image (PIL.Image)   : image input
        prediction (dict)   : 'infer' function output
        lw (int)            : bboxes linewidth
        fsz (int)           : font size
        offset (float|None) : amount of bounding boxes offset as percentage
    '''
    offsets = [offset] if offset else None
    boxes = np.array(prediction["bboxes"])
    probs = np.array(prediction["probs"])
    boxes = rescale_bboxes(boxes,image.size,offsets=offsets)
    texts = { 0:"image", 1:"drawing" }
    colors = {0:[1.0,0.0,0.0],1:[0.0,0.0,1.0]}
    plt.figure(figsize=(16,10))
    plt.imshow(image)
    ax = plt.gca()
    i = 0
    for p, (xmin, ymin, xmax, ymax) in zip(probs, boxes):
        cl = p.argmax()
        color = colors[cl]
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=color, linewidth=lw))
        text = f'{i} - {texts[cl]}: {p.max():0.2f}'
        ax.text(xmin, ymin, text, fontsize=fsz, bbox=dict(facecolor='yellow', alpha=0.5))
        i+=1
    plt.axis('off')
    plt.show()


##############
## Cropping ##
##############


def crop_documents_single(input, output_path, brain, threshold=0.75, thumbsize=190,
                          offset=0.01, conversion_dpi=300, compression_level=4,compression_wait=300):
    '''
    Crops PDF document into individual drawings and images

    Parameters:
        input (bytes) : byte stream of PDF file
        output_path (string) : path to output directory
        brain (dict) : ditionary containing fields ->
            model (Detr) : trained Detr model
            feature_extractor (DetrFeatureExtractor) : Detr feature extractor object from huggingface transformers
        threshold (float) : level of accepted confidence for output predictions
        thumbsize (int) : size for thumbnail
        offset (float) : offset percentage from actual bounding box prediction
        conversion_dpi (int) : pdf-to-image conversion resolution
        compression_level (int) : cropped pdf compression level (ghostscript)
        compression_wait (int) : seconds to wait for compression

    Returns
        (list) : list of 'process_document' function outputs
    '''
    model = brain["model"]
    feature_extractor = brain["feature_extractor"]
    logging.info("PROCESSING SINGLE INPUT\n")
    res = process_document(input, model, feature_extractor, threshold, conversion_dpi)
    crop_document(res, output_path, thumbsize, offset, compression_level,compression_wait)
    logging.info("Done")


def crop_documents(input_path, output_path, brain, threshold=0.75, thumbsize=190,
                   offset=0.01, conversion_dpi=300, compression_level=4,compression_wait=300):
    '''
    Crops PDF document into individual drawings and images

    Parameters:
        input_path (string) : path to PDF container directory
        output_path (string) : path to output directory
        brain (dict) : ditionary containing fields ->
            model (Detr) : trained Detr model
            feature_extractor (DetrFeatureExtractor) : Detr feature extractor object from huggingface transformers
        threshold (float) : level of accepted confidence for output predictions
        thumbsize (int) : size for thumbnail
        offset (float) : offset percentage from actual bounding box prediction
        conversion_dpi (int) : pdf-to-image conversion resolution
        compression_level (int) : cropped pdf compression level (ghostscript)
        compression_wait (int) : seconds to wait for compression

    Returns
        (list) : list of 'process_document' function outputs
    '''
    inputs = glob.glob(input_path+"*.pdf")
    model = brain["model"]
    feature_extractor = brain["feature_extractor"]
    logging.info("PROCESSING INPUTS\n")
    for j,path in enumerate(inputs):
        logging.info(f"Processing {j+1} / {len(inputs)} PDFs\n")
        res = process_document(path, model, feature_extractor, threshold, conversion_dpi)
        crop_document(res, output_path, thumbsize, offset, compression_level,compression_wait)
    logging.info("Done")


def process_document(pdf, model, feature_extractor, threshold=0.75, conversion_dpi=300):
    '''
    Predicts bounding boxes on a single document

    Parameters:
        pdf (string|bytes) : path or bytes stream of pdf file
        model (Detr) : trained Detr model
        feature_extractor (DetrFeatureExtractor) : Detr feature extractor object from huggingface transformers
        threshold (float) : level of accepted confidence for output predictions
        conversion_dpi (int) : pdf-to-image conversion resolution

    Returns
        (dict) : dictionary with fields ->
            predictions (list) : 'infer' function output
            path (string) : path of pdf file
    '''
    logging.info("\tConverting PDF to images...")
    images = convert_pdf_to_images(pdf, dpi=conversion_dpi)
    logging.info("\tRunning inference...")
    predictions = infer(images, model, feature_extractor, threshold)
    return {"predictions":predictions, "pdf":pdf}


def crop_document(processed_doc, output_path, thumbsize=190, offset=0.01, compression_level=4,compression_wait=300):
    '''
    Crops a single PDF document

    Parameters:
        processed_doc (dict) : 'process_document' function output
        output_path (string) : directory to save the croppped results
        thumbsize (int) : size for thumbnail
        offset (float) : offset percentage from actual bounding box prediction
        compression_level (int) : cropped pdf compression level (ghostscript)
        compression_wait (int) : seconds to wait for compression
    '''
    fs_structure = {"images"   : {"pdf":{"optimized":{},"original":{}}, "png":{}, "thumbs":{}},
                    "drawings" : {"pdf":{"optimized":{},"original":{}}, "png":{}, "thumbs":{}}}
    pdf = processed_doc["pdf"]
    predictions = processed_doc["predictions"]
    fs = create_file_structure(pdf, output_path, fs_structure)
    process_pages(pdf,predictions,fs,thumbsize,offset,compression_level,compression_wait)


def process_pages(pdf, predictions, fs, thumbsize=190, offset=0.01, compression_level=4, compression_wait=300 ):
    '''
    Crops a single PDF document - called by 'crop_document' function

    Parameters:
        pdf (string|bytes) : path or bytes stream of PDF file
        predictions (list) : list of 'process_document' function outputs
        fs (dict) : output of 'create_fs' function - to save results
        thumbsize (int) : size for thumbnail
        offset (float) : offset percentage from actual bounding box prediction
        compression_level (int) : cropped pdf compression level (ghostscript)
        compression_wait (int) : seconds to wait for compression
    '''
    open_doc = lambda f: fitz.open(f) if type(f)==str else fitz.open(stream=f, filetype="pdf")
    doc = open_doc(pdf)
    logging.info("\tExtracting indexed images...")
    indexed_bboxes = get_indexed_images(doc)
    num_pages = doc.page_count
    logging.info("\tProcessing pages...\n")
    for page_no in range(num_pages):
        logging.info(f"\t\tPage {page_no+1} / {num_pages}\n")
        page = doc[page_no]
        prediction = predictions[page_no]
        if len(prediction['bboxes']) > 0:
            process_page(page, prediction, indexed_bboxes[page_no], fs, offset, thumbsize, compression_level,compression_wait)
        else:
            logging.info("\t\t- nothing found in this page")
        doc.close()
        doc = open_doc(pdf)


@timer_print(indentation="\t\t")
def process_page(page,prediction,indexed_bboxes, fs, offset, thumbsize, compression_level,compression_wait):
    '''
    Crops a single PDF page - called by 'process_pages' function

    Parameters:
        page (fitz.Page) : pdf page
        prediction (dict) : 'infer' function output
        indexed_bboxes (4-n array) : 'get_indexed_images' function output
        fs (dict) : output of 'create_fs' function - to save results
        offset (float) : offset percentage from actual bounding box prediction
        thumbsize (int) : size for thumbnail
        compression_level (int) : cropped pdf compression level (ghostscript)
        compression_wait (int) : seconds to wait for compression
    '''
    if len(indexed_bboxes) > 0:
        prediction_bboxes = prediction['bboxes']
        aligned_bboxes, needs_offset = align_bboxes(indexed_bboxes,prediction_bboxes,0.9)
        prediction['bboxes'] = aligned_bboxes
    else:
        needs_offset = np.array(len(prediction['bboxes'])*[1])
    offsets = np.array(needs_offset)*offset
    process_bboxes(page, prediction, fs, offsets, thumbsize, compression_level,compression_wait)
    print("")



def process_bboxes(page, prediction, fs, offsets=None, thumbsize=190, compression_level=4, compression_wait=300):
    '''
    Processes and saves all bounding boxes in a single page - called by 'process_pages' function

    Parameters:
        page (fitz.Page) : pdf page
        prediction (dict) : 'infer' function output
        fs (dict) : output of 'create_fs' function - to save results
        offsets (list|None) : offset percentage for each box
        thumbsize (int) : size for thumbnail
        compression_level (int) : cropped pdf compression level (ghostscript)
        compression_wait (int) : seconds to wait for compression
    '''
    bboxes, labels = prediction["bboxes"], prediction["labels"]
    page_size = [*map(float,page.mediabox[2:])]
    page_bboxes = rescale_bboxes(bboxes, page_size, offsets)
    for i,bbox in enumerate(page_bboxes):
        logging.info(f"\t\t- processing area {i+1} / {len(page_bboxes)}")
        res = process_bbox(bbox, page, thumbsize)
        lbl = 'drawings' if labels[i]==1 else 'images'
        fname = f"{page.number+1}_{i+1}"
        save_cropped(res, fs[lbl], fname , compression_level, compression_wait)
        logging.info(f"\t\t\tsaved")
        res['drawing'].close()


def process_bbox(bbox, page, thumbsize):
    '''
    Processes a single bounding box from a single pdf page - called by 'process_bboxes' function

    Parameters:
        bbox (4-1 array) : coordinates of bounding box as a numpy array
        page (fitz.Page) : pdf page
        thumbsize (int) : size for thumbnail

    Returns:
        (dict) : dictionary with fields ->
            drawing (fitz.Document) : cropped PDF file
            image (PIL.Image) : cropped PDF file as image
            thumbnail (PIL.Image) : cropped PDF file as thumbnail image
    '''
    drawing = crop_drawing(bbox, page)
    image = image_from_pdf_page(drawing[0],dpi=300)
    thumbnail = make_thumbnail(image,(thumbsize,thumbsize))
    res = {"drawing":drawing, "image":image, "thumbnail":thumbnail}
    return res


def crop_drawing(bbox, page):
    '''
    Crops a single bounding box from a single pdf page as PDF - called by 'process_bbox' function

    Parameters:
        bbox (4-1 array) : coordinates of bounding box as a numpy array
        page (fitz.Page) : pdf page
        thumbsize (int) : size for thumbnail

    Returns:
        (fitz.Document) : cropped PDF file
    '''
    out = fitz.open()
    rect = fitz.Rect(bbox.tolist())
    newpage = out.new_page(-1, rect.width, rect.height)
    newpage.show_pdf_page(newpage.rect, page.parent, page.number, clip=rect)
    return out


def make_thumbnail(image, size=(190,190), bg_color=(255, 255, 255, 0)):
    '''
    Creates a thumbnail from an image

    Parameters:
        image (PIL.Image) : image
        size (tuple) : thumbnail size
        bg_color (tuple) : background color

    Returns:
        (PIL.Image) : thumbnail image
    '''
    thumbnail = copy.deepcopy(image)
    thumbnail.thumbnail(size, Image.ANTIALIAS)
    background = Image.new('RGBA', size, bg_color)
    background.paste(thumbnail, (int((size[0] - thumbnail.size[0]) / 2), int((size[1] - thumbnail.size[1]) / 2)))
    return background


def save_cropped(crop, dst, name, compression_level=4, wait=300):
    '''
    Saves a cropped object into a file structure - called by 'process_bboxes' function

    Parameters:
        crop (dict) : output of 'process_bbox' function
        dst (string) : path
        name (string) : name for the new file
        compression_level (int) : cropped pdf compression level (ghostscript)
        save_original_pdf (bool) : whether to save the croppped pdf in original size too
        wait (int) : seconds to wait for compression
    '''
    fname_img = f"{dst}/png/element_{name}.png"
    fname_thb = f"{dst}/thumbs/thumbnail_{name}.png"
    fname_drw = f"{dst}/pdf/original/element_{name}.pdf"
    fname_cmp = f"{dst}/pdf/optimized/element_{name}.pdf"
    crop['image'].save( fname_img, dpi=(300, 300) )
    crop['thumbnail'].save( fname_thb, dpi=(100, 100))
    crop['drawing'].save( fname_drw )
    try_to_compress( wait, compress_pdf, fname_drw, fname_cmp, compression_level )


def rescale_bboxes(out_bbox, size, offsets=None):
    '''
    Rescales bounding boxes (from relative to absolute size)

    Parameters:
        out_bbox (4-n array) : array of bounding boxes
        size (tuple) : target absolute dimensions
        offsets (list|None) : offset percentage for each box

    Returns:
        (4-n array) : rescaled and offset bounding boxes
    '''
    img_w, img_h = size
    b = out_bbox * np.array([img_w, img_h, img_w, img_h], dtype="float32").reshape(1,-1)
    if str(offsets)!='None':
        b = offset_bboxes(b, size, offsets)
    return b


def offset_bboxes(boxes, bound, offsets):
    '''
    Offsets bounding boxes by defined percentage

    Parameters:
        boxes (4-n array) : array of bounding boxes
        bound (tuple) : size boundaries
        offsets (list) : offset percentage for each box

    Returns:
        (4-n array) : offset bounding boxes
    '''
    bound_x, bound_y = bound
    offsets = np.array(offsets)
    off = (np.mean(np.array([np.diff(boxes[:,[0,2]]).flatten(), np.diff(boxes[:,[1,3]]).flatten()]),0)*offsets).reshape(-1,1)
    new = np.concatenate((boxes[:,:2] - off, boxes[:,2:] + off),1)
    new_x = new[:,[0,2]]
    new_y = new[:,[1,3]]
    new_x[new_x>bound_x] = bound_x-1
    new_y[new_y>bound_y] = bound_y-1
    new_x[new_x<0] = 0
    new_y[new_y<0] = 0
    new_bboxes = np.array([new_x[:,0], new_y[:,0], new_x[:,1], new_y[:,1]]).T
    return new_bboxes


@timer_print(indentation="\t")
def get_indexed_images(doc):
    '''
    Saves indexed images from a PDF file

    Parameters:
        doc (fitz.Document) : PDF file

    Returns:
        (4-n array) : bounding boxes of extratced images
    '''
    bboxes = {}
    for i,page in enumerate(doc.pages()):
        bboxes[i] = []
        page_size = [*page.mediabox_size]
        imgs = page.get_image_info(xrefs=True)
        _bboxes = np.array([*map(lambda x: x['bbox'], imgs)])
        if len(_bboxes)>0:
            bboxes[i] = np.unique(np.round(_bboxes,2),axis=0) / (page_size*2)
    return bboxes


def align_bboxes(indexed_bboxes, predicted_bboxes, threshold = 0.9):
    '''
    Replaces predicted bounding box with indexed bounding box if their is IOU > threshold

    Parameters:
        indexed_bboxes (4-n array) : bounding boxes to test against
        predicted_bboxes (4-k array) : bounding boxes to be tested (and replaced)
        threshold (float) : intersection over union (iou) threshold

    Returns:
        (4-n array) : aligned bounding boxes
        (1-n array) : boolean array of whether bounding box was replaced or not (1:no, 0:yes)
    '''
    aligned_boxes, not_switched = [], []
    for box_p in predicted_bboxes:
        p_ious =[*map(lambda b: iou(b,box_p), indexed_bboxes)]
        max_i, argmax_i = np.max(p_ious), np.argmax(p_ious)
        new_box, nswitch = [box_p, 1] if max_i<threshold else [indexed_bboxes[argmax_i], 0]
        not_switched.append(nswitch)
        aligned_boxes.append(new_box)
    return np.array(aligned_boxes), np.array(not_switched)


def iou(box1, box2):
    '''
    Computes intersection over union (iou) between two areas

    Parameters:
        bbox1 (4-n array) : area 1
        bbox2 (4-n array) : area 2

    Returns:
        (float) : iou percentage [0.0-1.0]
    '''
    zipped_box = np.array([box1,box2]).T
    x1, y1 = zipped_box[:2].max(1)
    x2, y2 = zipped_box[2:].min(1)
    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou


@timer_print(indentation="\t")
def convert_pdf_to_images(pdf,dpi=300):
    '''
    Convert a pdf file to a list of images.

    Parameters:
        pdf (string|bytes) : path or byte stream of the pdf file
        dpi (int) : image dpi resolution

    Returns:
        ([PIL.Image]) : list of images
    '''
    images = []
    doc = fitz.open(pdf) if type(pdf)==str else fitz.open(stream=pdf, filetype="pdf")
    for page in doc.pages():
        img = image_from_pdf_page(page,dpi=dpi)
        images.append(img)
    return images


def image_from_pdf_page(page,**args):
    '''
    Transforms a PDF page to an image

    Parameters:
        page (fitz.Page) : pdf page

    Returns:
        (PIL.Image) : image
    '''
    return Image.open(io.BytesIO(page.get_pixmap(**args).tobytes()))


def compress_pdf(input_file_path, output_file_path, power=0):
    '''
    Function to compress PDF via Ghostscript command line interface.
    From: https://github.com/theeko74/pdfc

    Parameters:
        input_file_path (string) : input path
        output_file_path (string) : output path
        power (int) : type of compression
    '''
    quality = {0: '/default',1: '/prepress',2: '/printer',3: '/ebook',4: '/screen'}
    gs = get_ghostscript_path()
    cores = multiprocessing.cpu_count()
    initial_size = os.path.getsize(input_file_path)
    subprocess.call([gs, '-sDEVICE=pdfwrite', '-dCompatibilityLevel=1.4',
                    '-dPDFSETTINGS={}'.format(quality[power]),
                    '-dNOPAUSE', '-dQUIET', '-dBATCH',
                    '-sOutputFile={}'.format(output_file_path),
                    '-dNumRenderingThreads={}'.format(cores), #
                    '-dNOGC', #
                    '-c "60000000 setvmthreshold"', #
                    '-dBufferSpace=2000000000', #
                    '-f', input_file_path] #
    )


def get_ghostscript_path():
    '''
    Find Ghostscript executable path.
    From: https://github.com/theeko74/pdfc
    '''
    gs_names = ['gs', 'gswin32', 'gswin64']
    for name in gs_names:
        if shutil.which(name):
            return shutil.which(name)
    raise FileNotFoundError(f'No GhostScript executable was found on path ({"/".join(gs_names)})')


def create_file_structure(pdf, target, struct):
    '''
    Creates the output file structure

    Parameters:
        pdf (string|bytes) : path or bytes stream of PDF file (its name will be the name of the output dir)
        target (string) : directory path for output
        struct (dict) : desired file structure

    Returns:
        (dict) : dictionary containing all the created paths as values
    '''
    name = pdf.split("/")[-1].split(".")[0] if type(pdf)==str else ''
    root_path = f"{target}/{name}"
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    res = create_fs([struct],[root_path], root_path, [])
    return res


def create_fs(dicts, roots, parent, paths=[]):
    '''
    Creates an arbitrary file structure recusrively

    Parameters:
        dicts (list) : list of dicts that indicate the file structure ->
                [ { "a1" : {},
                    "a2" : { "b1" : {},
                             "b2" : {"c1":{}}},
                    "a3" : { "b1" : {} }} ]
        roots (list) : root of each top-level element in dicts (eg. ['.'])
        parent (string) : parent directory
        paths (list) : list used for recursion

    Returns:
        (dict) : dictionary containing all the created paths as values
    '''
    new_dicts = []
    new_roots = []
    for d,r in zip(dicts,roots):
        keys = [*d.keys()]
        for k in keys:
            new_dicts.append(d[k])
            new_roots.append(f"{r}/{k}")
            paths.append(f"{r}/{k}")
    if len(new_dicts)>0:
        return create_fs(new_dicts, new_roots, parent, paths)
    else:
        _ = [*map(lambda p: os.mkdir(p) if not os.path.exists(p) else None, paths)]
        paths_dict = dict(map(lambda s: [s.replace(parent+"/","").replace("/","_"), s + "/"], paths))
        return paths_dict        

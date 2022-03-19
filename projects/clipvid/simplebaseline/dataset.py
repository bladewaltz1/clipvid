import functools
import os
import random
import pickle
import xml.etree.ElementTree as ET

from detectron2.data import DatasetCatalog, detection_utils
from detectron2.structures import BoxMode


classes_map = ['n02691156', 'n02419796', 'n02131653', 'n02834778',
               'n01503061', 'n02924116', 'n02958343', 'n02402425',
               'n02084071', 'n02121808', 'n02503517', 'n02118333',
               'n02510455', 'n02342885', 'n02374451', 'n02129165',
               'n01674464', 'n02484322', 'n03790512', 'n02324045',
               'n02509815', 'n02411705', 'n01726692', 'n02355227',
               'n02129604', 'n04468005', 'n01662784', 'n04530566',
               'n02062744', 'n02391049']


def get_vid_train_dicts(data_dir, num_chunks):
    dump_file = os.path.join(data_dir, f"vid_train_{num_chunks}chunks.pkl")
    if os.path.exists(dump_file):
        database = pickle.load(open(dump_file, "rb"))
        database = database * num_chunks
        return database

    database = []
    vid_dir = os.path.join(data_dir, "%s", "VID", "train2")

    for video in sorted(os.listdir(vid_dir % "Data")):
        jpegs = sorted(os.listdir(os.path.join(vid_dir % "Data", video)))
        xmls = sorted(os.listdir(os.path.join(vid_dir % "Annotations", video)))
        assert len(jpegs) == len(xmls)

        records = []
        for jpeg, xml in zip(jpegs, xmls):
            assert os.path.splitext(jpeg)[0] == os.path.splitext(xml)[0]

            record = {}
            record["filename"] = os.path.join(vid_dir % "Data", video, jpeg)
            tree = ET.parse(os.path.join(vid_dir % "Annotations", video, xml))
            size = tree.find("size")
            record["height"] = int(size.find("height").text)
            record["width"] = int(size.find("width").text)

            annos = []
            objects = tree.findall("object")
            for obj in objects:
                anno = {}
                name = obj.find("name").text.lower().strip()
                anno["category_id"] = classes_map.index(name)

                bbox = obj.find("bndbox")
                anno["bbox"] = [
                    float(bbox.find("xmin").text),
                    float(bbox.find("ymin").text),
                    float(bbox.find("xmax").text),
                    float(bbox.find("ymax").text)
                ]

                anno["bbox_mode"] = BoxMode.XYXY_ABS
                annos.append(anno)

            if len(annos) > 0:
                record["annotations"] = annos
                records.append(record)

        if len(records) > 0:
            chunks = split_train_video(records, num_chunks)
            database.append(chunks)

    pickle.dump(database, open(dump_file, "wb"))
    database = database * num_chunks
    return database


def split_train_video(records, num_chunks):
    chunks = []
    if len(records) <= num_chunks:
        for i in range(len(records)):
            chunks.append([records[i]])
        return chunks

    chunk_size = len(records) // num_chunks
    for i in range(num_chunks):
        chunks.append(records[i * chunk_size : (i + 1) * chunk_size])
    if len(records) > chunk_size * num_chunks:
        chunks.append(records[chunk_size * num_chunks : len(records)])
    return chunks


def get_vid_val_dicts(data_dir, chunk_size):
    count = 0
    database = []
    vid_dir = os.path.join(data_dir, "%s", "VID", "val")

    for video in sorted(os.listdir(vid_dir % "Data")):
        jpegs = sorted(os.listdir(os.path.join(vid_dir % "Data", video)))
        xmls = sorted(os.listdir(os.path.join(vid_dir % "Annotations", video)))
        assert len(jpegs) == len(xmls)

        records = []
        for jpeg, xml in zip(jpegs, xmls):
            assert os.path.splitext(jpeg)[0] == os.path.splitext(xml)[0]

            record = {}
            record["filename"] = os.path.join(vid_dir % "Data", video, jpeg)
            tree = ET.parse(os.path.join(vid_dir % "Annotations", video, xml))
            size = tree.find("size")
            record["height"] = int(size.find("height").text)
            record["width"] = int(size.find("width").text)
            record["data_index"] = count
            count += 1
            records.append(record)

        chunks = split_val_video(records, chunk_size)
        database.extend(chunks)

    return database


def split_val_video(records, chunk_size):
    chunks = []
    if len(records) <= chunk_size:
        records[-1]["last_frame"] = 1
        records[0]["first_frame"] = 1
        return [records]

    random.shuffle(records) # NOTE

    num_chunks = len(records) // chunk_size
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunks.append(records[start : end])

    if end < len(records):
        chunks.append(records[end:])

    chunks[-1][-1]["last_frame"] = 1
    chunks[0][0]["first_frame"] = 1
    return chunks


def get_det_dicts(data_dir, det_data_file):
    dump_file = os.path.join(data_dir, f"meta_det.pkl")
    if os.path.exists(dump_file):
        database = pickle.load(open(dump_file, "rb"))
        return database

    database = []
    det_data_handler = open(os.path.join(data_dir, det_data_file))
    data_dir = os.path.join(data_dir, "%s", "DET")
    jpegs_dir = os.path.join(data_dir % "Data")
    xmls_dir = os.path.join(data_dir % "Annotations")

    for line in det_data_handler.readlines():
        name = line.split(' ')[0]
        jpeg = name + '.JPEG'
        xml = name + '.xml'

        record = {}
        record["filename"] = os.path.join(jpegs_dir, jpeg)
        tree = ET.parse(os.path.join(xmls_dir, xml))
        size = tree.find("size")
        record["height"] = int(size.find("height").text)
        record["width"] = int(size.find("width").text)

        annos = []
        objects = tree.findall("object")
        for obj in objects:
            anno = {}
            name = obj.find("name").text.lower().strip()
            if name not in classes_map:
                continue
            anno["category_id"] = classes_map.index(name)

            bbox = obj.find("bndbox")
            anno["bbox"] = [
                float(bbox.find("xmin").text),
                float(bbox.find("ymin").text),
                float(bbox.find("xmax").text),
                float(bbox.find("ymax").text)
            ]

            anno["bbox_mode"] = BoxMode.XYXY_ABS
            annos.append(anno)

        image = detection_utils.read_image(record["filename"], "RGB")
        image_wh = (image.shape[1], image.shape[0])
        expected_wh = (record["width"], record["height"])
        if image_wh == expected_wh:
            if len(annos) > 0:
                record["annotations"] = annos
                database.append([record])

    pickle.dump(database, open(dump_file, "wb"))
    return database


def register_vid_instances(data_root, 
                           num_train_chunks=15, 
                           val_chunk_size=5):
    """
    [ # database
      [ # records of one chunk
        { # record of one frame
          "filename": str, 
          "height": int, 
          "width": int, 
          "data_index": int (optional),
          "annotations": [
            { # record of one instance
              "bbox": [float,float,float,float],
              "category_id": int,
              "bbox_mode": XYWH_ABS,
            },{},...
          ]
        },{},...
      ],[],...
    ]
    """
    path = os.path.join(data_root, "ILSVRC2015")

    DatasetCatalog.register(f"vid_train_{num_train_chunks}chunks",
        lambda: get_vid_train_dicts(path, num_train_chunks))

    DatasetCatalog.register(f"vid_val_{val_chunk_size}csize", 
        functools.partial(get_vid_val_dicts, path, val_chunk_size)
    )

    DatasetCatalog.register("det", lambda: get_det_dicts(path,
        "ImageSets/DET_train_30classes.txt"))

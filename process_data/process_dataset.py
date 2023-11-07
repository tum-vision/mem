import os
import numpy as np
import multiprocessing
import configargparse
import struct
import h5py
from utils import *

dt = np.dtype('int,int,int,int')

def get_args():
    parser = configargparse.ArgumentParser('N-Caltech101 processing', add_help=False)

    parser.add_argument('--dataset', required=True, type=str, choices=["ncaltech101", "ncars", "nimagenet", "dsec"], help="The dataset to be processed.")
    parser.add_argument('--input', required=True, type=str, help="folder with the event dataset.")
    parser.add_argument('--output', required=True, type=str, help="folder to save processed dataset")
    parser.add_argument('--cores', default=1, type=int, help="the amount of processes to be started")
    parser.add_argument('--split', type=str, help="Split file")
    parser.add_argument('--class_folder', type=str, help="Class to be processed. If None, all classes are processed.")

    return parser.parse_args()


def ncaltech101(folder, args):
    val_set = []
    if args.split is not None:
        with open(args.split) as f:
            folder_set = filter(lambda x: folder in x, f.readlines())
            val_set = list(map(lambda x: x.split("/")[-1][:-5].strip(), filter(lambda x: "val" in x, folder_set)))
            train_set = list(map(lambda x: x.split("/")[-1][:-5].strip(), filter(lambda x: "train" in x, folder_set)))

    for filename in os.listdir(os.path.join(args.input, folder)):
        # check if file in split
        if filename.split(".")[0] in train_set:
            data_split = "train"
        elif filename.split(".")[0] in val_set:
            data_split = "val"
        else:
            continue

        if not os.path.exists(os.path.join(args.output, data_split, folder)):
            os.makedirs(os.path.join(args.output, data_split, folder))

        events = []

        print(f"{folder}/{data_split}/{filename}")

        with open(os.path.join(args.input, folder, filename), "rb") as file:
            while True:
                data = file.read(5)
                if not data:
                    break
                y = data[0]
                x = data[1]
                p = (data[2] >> 7) & 0x01
                t = (data[2] & 0x7f).to_bytes(1,byteorder='big') + data[3:5]
                t = int.from_bytes(t, byteorder='big')
                
                p = 2.*p-1.
                events.append([float(y),float(x),float(t),float(p)])
        
        events = np.array(events).astype(float)
        np.save(os.path.join(args.output, data_split, folder, filename.split(".")[0]+".npy"), events)


def ncars(folder, args):
    for data_split in ["train", "val"]:
        split_name = "n-cars_train" if data_split == "train" else "n-cars_test"

        for filename in os.listdir(os.path.join(args.input, split_name, folder)):
            if not os.path.exists(os.path.join(args.output, data_split, folder)):
                os.makedirs(os.path.join(args.output, data_split, folder))

            events = []

            print(f"{folder}/{data_split}/{filename}")

            with open(os.path.join(args.input, split_name, folder, filename), "rb") as file:
                while True:
                    pointer = file.tell()
                    data = file.readline(256)
                    if data[0] != 37:
                        file.seek(pointer)
                        break

                file.read(2)

                while True:
                    raw_timestamp = file.read(4)
                    raw_data = file.read(4)

                    if not raw_data and not raw_timestamp:
                        break

                    t = struct.unpack('I', raw_timestamp)[0]
                    data = int.from_bytes(raw_data, byteorder='little')
                    
                    y = (data & 0x00003fff)
                    x = (data & 0x0fffc000) >> 14
                    p = (data & 0x10000000) >> 28
                    
                    events.append([y,x,t,bool(p)])
            
            events = np.array(events).astype(float)
            np.save(os.path.join(args.output, data_split, folder, filename.split(".")[0]), events)


def nimagenet(folder, args):
    for data_split in ["train", "val"]:
        split_name = "extracted_train" if data_split == "train" else "extracted_val"

        for filename in os.listdir(os.path.join(args.input, split_name, folder)):
            if not os.path.exists(os.path.join(args.output, data_split, folder)):
                os.makedirs(os.path.join(args.output, data_split, folder))

            data = np.load(os.path.join(args.input, split_name, folder, filename))["event_data"]
            np.save(os.path.join(args.output, data_split, folder, filename.split(".")[0]+".npy"), data)


def dsec(folder, args):
    STEREO = "right"
    H, W = 480, 640

    for data_split in ["train", "val"]:
        split_name = "train_events" if data_split == "train" else "test_events"

        for j, seq in enumerate(folder):
            if os.path.isfile(os.path.join(args.input, split_name, seq)) or ".py" in seq or ".txt" in seq or ".csv" in seq or ".png" in seq or ".jpg" in seq or ".npy" in seq:
                print(f"Skipping {os.path.join(args.input, split_name, seq)} ")
                continue

            datapath = os.path.join(args.input, split_name, seq, "events", STEREO)
            if not os.path.exists(datapath):
                print(f"**** Big Warning: Did not find events in {datapath} for seq {seq}")
            
            outfolder = os.path.join(args.output, split_name, seq, f"events_{STEREO}_npy")
            os.makedirs(outfolder, exist_ok=True)

            evfile = h5py.File(f"{datapath}"+"/events.h5", "r")
            event_slicer = EventSlicer(evfile)
            dt_us = evfile["events"]["t"][-1]-evfile["events"]["t"][0]
            t_offset = int(np.array(evfile["t_offset"]))
            tss_us = np.linspace(evfile["events"]["t"][0], evfile["events"]["t"][-1], int(dt_us*1e-6*20), dtype=np.int64)[1:] + t_offset
            dt_avg_us = np.diff(tss_us).mean()

            for i, ts in enumerate(tss_us):
                ts0_us = ts - dt_avg_us
                ts1_us = ts0_us + dt_avg_us
                slice = event_slicer.get_events(ts0_us, ts1_us)

                if slice is None:
                    print(f"Warning: Found no-events in step {i}/{len(tss_us)} of seq {seq} from {ts0_us/1e3} to {ts1_us/1e3} milisecs.")
                    continue

                assert np.all(slice["x"] >= 0)
                assert np.all(slice["y"] >= 0)
                assert np.all(slice["t"] >= 0)
                assert np.all(slice["x"] < W)
                assert np.all(slice["y"] < H)

                if i > 0 and i < len(tss_us)-2:
                    if not (np.abs(slice["t"][-1]-ts1_us) <= 2500):
                        print(f"Warning: {np.abs(slice['t'][-1]-ts1_us)}")
                    if not (np.abs(slice["t"][0]-ts0_us)) <= 2500:
                        print(f"Warning: {np.abs(slice['t'][0]-ts0_us)}")

                ev_batch = np.stack((slice["x"], slice["y"], slice["y"]*0, slice["p"])).T # (N, 4)
                np.save(os.path.join(outfolder, f"{i:06d}.npy"), ev_batch)

            evfile.close()


def convert_folders(convert_function, folders, args):
    for folder in folders:
        convert_function(folder, args)


if __name__ == "__main__":
    args = get_args()

    if args.dataset == "ncaltech101" and (args.split is None or not os.path.exists(args.split)):
        print("WARNING: no split file specified or path invalid. All data will be used for training.")

    convert_function = ""
    if args.dataset == "ncaltech101":
        convert_function = ncaltech101
    elif args.dataset == "ncars":
        convert_function = ncars
    elif args.dataset == "nimagenet":
        convert_function = nimagenet
    elif args.dataset == "dsec":
        convert_function = dsec
    else:
        raise ValueError("Unknown dataset")
        

    if args.class_folder is not None:
        convert_function(args.class_folder, args)
    elif args.cores == 1:
        for folder in os.listdir(args.input):
            convert_function(folder, args)
    else:
        processes = []
        folders = os.listdir(args.input)
        if args.dataset == "ncars":
            folders = os.listdir(os.path.join(args.input, "n-cars_train"))
        if args.dataset == "nimagenet":
            folders = os.listdir(os.path.join(args.input, "extracted_train"))

        for i in range(args.cores):
            p = multiprocessing.Process(target=convert_folders, args=(convert_function, folders[i::args.cores],args,))
            processes.append(p)
            p.start()

        for process in processes:
           process.join()

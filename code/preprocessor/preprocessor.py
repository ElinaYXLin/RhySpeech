import os
import random
import json

import tgt
import librosa
import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import audio as Audio


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.rhy_out_dir = config["path"]["rhy_preprocessed_path"]
        self.val_size = config["preprocessing"]["val_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]

        assert config["preprocessing"]["pitch"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        assert config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        self.pitch_phoneme_averaging = (
            config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )
        self.energy_phoneme_averaging = (
            config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )

        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]
        self.pause_normalization = config["preprocessing"]["pause"]["normalization"]

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    def rhy_build_from_path(self):
        os.makedirs((os.path.join(self.rhy_out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.rhy_out_dir, "pitch")), exist_ok=True)
        os.makedirs((os.path.join(self.rhy_out_dir, "energy")), exist_ok=True)
        os.makedirs((os.path.join(self.rhy_out_dir, "duration")), exist_ok=True)
        os.makedirs((os.path.join(self.rhy_out_dir, "pause")), exist_ok=True)

        print("Processing Data ...")
        out = list()
        n_frames = 0
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()
        pause_scaler = StandardScaler()
        prog = 0

        # Compute pitch, energy, duration, and mel-spectrogram
        speakers = {}
        speakerLabel = 0
        for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))):
            if (speaker == '.DS_Store'):
                continue


            for chapter in os.listdir(os.path.join(self.in_dir, speaker)):
                if (chapter == '.DS_Store'):
                    continue

                for wav_name in os.listdir(os.path.join(self.in_dir, speaker, chapter)):

                    if ".wav" not in wav_name:
                        continue

                

                    basename = wav_name.split(".")[0]
                    tg_path = os.path.join(
                        self.out_dir, "TextGrids", speaker, chapter, "{}.TextGrid".format(basename)
                    )
                    if os.path.exists(tg_path):
                        ret = self.rhy_process_utterance(speaker, chapter, basename, 4)
                        if ret is None:
                            continue
                        else:
                            info, pitch, energy, pause, n = ret
                        out.append(info)

                        pause = np.asarray(pause)

                        if len(pitch) > 0:
                            pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                        if len(energy) > 0:
                            energy_scaler.partial_fit(energy.reshape((-1, 1)))
                        if len(pause) > 0:
                            pause_scaler.partial_fit(pause.reshape((-1, 1)))

                        n_frames += n
                    else:
                        print("N/A")

            speakers[speaker] = speakerLabel
            speakerLabel += 1

        print("Computing statistic quantities ...")
        # Perform normalization if necessary
        if self.pitch_normalization:
            pitch_mean = pitch_scaler.mean_[0]
            pitch_std = pitch_scaler.scale_[0]
        else:
            # A numerical trick to avoid normalization...
            pitch_mean = 0
            pitch_std = 1
        if self.energy_normalization:
            energy_mean = energy_scaler.mean_[0]
            energy_std = energy_scaler.scale_[0]
        else:
            energy_mean = 0
            energy_std = 1

        if self.pause_normalization:
            pause_mean = pause_scaler.mean_[0]
            pause_std = pause_scaler.scale_[0]
        else:
            pause_mean = 0
            pause_std = 1

        pitch_min, pitch_max = self.normalize(
            os.path.join(self.rhy_out_dir, "pitch"), pitch_mean, pitch_std
        )
        energy_min, energy_max = self.normalize(
            os.path.join(self.rhy_out_dir, "energy"), energy_mean, energy_std
        )
        pause_min, pause_max = self.normalize(
            os.path.join(self.rhy_out_dir, "pause"), pause_mean, pause_std
        )

        # Save files
        with open(os.path.join(self.rhy_out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        with open(os.path.join(self.rhy_out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch": [
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
                ],
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
                ],
                "pause": [
                    float(pause_min),
                    float(pause_max),
                    float(pause_mean),
                    float(pause_std),
                ]
            }
            f.write(json.dumps(stats))

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        random.shuffle(out)
        out = [r for r in out if r is not None]

        # Write metadata
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size :]:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[: self.val_size]:
                f.write(m + "\n")

        return out

    def rhy_conc_build_from_path(self):
        os.makedirs((os.path.join(self.rhy_out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.rhy_out_dir, "pitch")), exist_ok=True)
        os.makedirs((os.path.join(self.rhy_out_dir, "energy")), exist_ok=True)
        os.makedirs((os.path.join(self.rhy_out_dir, "duration")), exist_ok=True)
        os.makedirs((os.path.join(self.rhy_out_dir, "pause")), exist_ok=True)

        print("Processing Data ...")
        out = list()
        n_frames = 0
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()
        pause_scaler = StandardScaler()
        prog = 0

        # Compute pitch, energy, duration, and mel-spectrogram
        speakers = {}
        speakerLabel = 0
        for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))):
            if (speaker == '.DS_Store'):
                continue


            for chapter in os.listdir(os.path.join(self.in_dir, speaker)):
                if (chapter == '.DS_Store'):
                    continue
                
                outTemp = []

                for wav_name in os.listdir(os.path.join(self.in_dir, speaker, chapter)):

                    if ".wav" not in wav_name:
                        continue

                

                    basename = wav_name.split(".")[0]
                    tg_path = os.path.join(
                        self.out_dir, "TextGrids", speaker, chapter, "{}.TextGrid".format(basename)
                    )
                    if os.path.exists(tg_path):
                        ret = self.rhy_process_utterance(speaker, chapter, basename, 0.2)
                        if ret is None:
                            continue
                        else:
                            info, pitch, energy, pause, n = ret
                        outTemp.append(info)

                        pause = np.asarray(pause)

                        if len(pitch) > 0:
                            pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                        if len(energy) > 0:
                            energy_scaler.partial_fit(energy.reshape((-1, 1)))
                        if len(pause) > 0:
                            pause_scaler.partial_fit(pause.reshape((-1, 1)))

                        n_frames += n
                    else:
                        print("N/A")

                if (len(outTemp) >= 75):
                    for info in outTemp:
                        out.append(info)

            speakers[speaker] = speakerLabel
            speakerLabel += 1

        print("Computing statistic quantities ...")
        # Perform normalization if necessary
        if self.pitch_normalization:
            pitch_mean = pitch_scaler.mean_[0]
            pitch_std = pitch_scaler.scale_[0]
        else:
            # A numerical trick to avoid normalization...
            pitch_mean = 0
            pitch_std = 1
        if self.energy_normalization:
            energy_mean = energy_scaler.mean_[0]
            energy_std = energy_scaler.scale_[0]
        else:
            energy_mean = 0
            energy_std = 1

        if self.pause_normalization:
            pause_mean = pause_scaler.mean_[0]
            pause_std = pause_scaler.scale_[0]
        else:
            pause_mean = 0
            pause_std = 1

        pitch_min, pitch_max = self.normalize(
            os.path.join(self.rhy_out_dir, "pitch"), pitch_mean, pitch_std
        )
        energy_min, energy_max = self.normalize(
            os.path.join(self.rhy_out_dir, "energy"), energy_mean, energy_std
        )
        pause_min, pause_max = self.normalize(
            os.path.join(self.rhy_out_dir, "pause"), pause_mean, pause_std
        )

        # Save files
        with open(os.path.join(self.rhy_out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        with open(os.path.join(self.rhy_out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch": [
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
                ],
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
                ],
                "pause": [
                    float(pause_min),
                    float(pause_max),
                    float(pause_mean),
                    float(pause_std),
                ]
            }
            f.write(json.dumps(stats))

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        random.shuffle(out)
        out = [r for r in out if r is not None]

        # Write metadata
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size :]:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[: self.val_size]:
                f.write(m + "\n")

        return out

    def rhy_process_utterance(self, speaker, chapter, basename, threshold):
        wav_path = os.path.join(self.in_dir, speaker, chapter, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker, chapter, "{}.lab".format(basename))
        tg_path = os.path.join(
            self.out_dir, "TextGrids", speaker, chapter, "{}.TextGrid".format(basename)
        )

        # Get alignments
        textgrid = tgt.io.read_textgrid(tg_path)
        phone, pause, duration, start, end = self.get_alignment(
            textgrid.get_tier_by_name("phones")
        )

        numPauses = 0
        for instance in pause:
            numPauses += (instance > 0)

        fullDur = end-start
        if (numPauses < threshold/10*fullDur or sum(pause) < threshold*fullDur): # for rhythm-rich, is 0.5
            return None

        text = "{" + " ".join(phone) + "}"
        if start >= end:
            return None

        # Read and trim wav files
        wav, _ = librosa.load(wav_path)
        wav = wav[
            int(self.sampling_rate * start) : int(self.sampling_rate * end)
        ].astype(np.float32)

        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")

        # Compute fundamental frequency
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

        pitch = pitch[: sum(duration)]
        if np.sum(pitch != 0) <= 1:
            return None

        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]
        energy = energy[: sum(duration)]

        if (len(duration) != len(pause)):
            print(len(duration))
            print(len(pause))
            print(len(phone))
            print("Error")

        if self.pitch_phoneme_averaging:
            # perform linear interpolation
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch = interp_fn(np.arange(0, len(pitch)))

            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    pitch[i] = np.mean(pitch[pos : pos + d])
                else:
                    pitch[i] = 0
                pos += d
            pitch = pitch[: len(duration)]

        if self.energy_phoneme_averaging:
            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    energy[i] = np.mean(energy[pos : pos + d])
                else:
                    energy[i] = 0
                pos += d
            energy = energy[: len(duration)]

        # Save files
        dur_filename = "{}-duration-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.rhy_out_dir, "duration", dur_filename), duration)
        if (speaker == '2929'):
            print("Here")

        pause_filename = "{}-pause-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.rhy_out_dir, "pause", pause_filename), pause)

        pitch_filename = "{}-pitch-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.rhy_out_dir, "pitch", pitch_filename), pitch)

        energy_filename = "{}-energy-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.rhy_out_dir, "energy", energy_filename), energy)

        mel_filename = "{}-mel-{}.npy".format(speaker, basename)
        np.save(
            os.path.join(self.rhy_out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )
   
        return (
            "|".join([basename, speaker, text, raw_text]),
            self.remove_outlier(pitch),
            self.remove_outlier(energy),
        #    self.remove_outlier(pause),
            pause,
            mel_spectrogram.shape[1],
        )

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn", ""]

        phones = []
        durations = []
        pause = []
        start_time = 0
        end_time = 0
        end_idx = 0
        cnt = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s
        
            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)

                if (cnt != len(tier._objects)-1):
                    pause.append(np.round((tier._objects[cnt+1].start_time - tier._objects[cnt].end_time)* self.sampling_rate / self.hop_length))
                else:
                    pause.append(0)

                # if (cnt!=len(tier._objects)-1):
                #     if (tier._objects[cnt+1].text not in sil_phones):
                #         pause.append(0)
                #     else:
                #         pause.append(int(
                #             np.round(tier._objects[cnt+1].end_time * self.sampling_rate / self.hop_length)
                #             - np.round(tier._objects[cnt+1].start_time * self.sampling_rate / self.hop_length)
                #         )
                #         )
                # else:
                #     pause.append(0)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )
            cnt += 1

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, pause, durations, start_time, end_time

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value

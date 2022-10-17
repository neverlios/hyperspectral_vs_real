from dataclasses import dataclass
from multiprocessing.connection import wait
from typing import List, Tuple
import numpy as np
import awb
import argparse
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
from pathlib import Path
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial
import json

wv_hsi = np.linspace(400, 730, num=34, endpoint=True)

default_parameters = dict(
    dsize=512 // 4,  # resizing to 128 x 128 x 34
    recalc=False,  # if True recalculate all results
    illum_names=None,
    debug=False  # if True save debug data (like illustrations and etc)
)

def parse_args():
    parser = argparse.ArgumentParser(
        'Simulate data from Li HSI reflectances dataset')
    parser.add_argument('-h5', '--h5_path', type=Path,
                        default='../../../../../hyperspectral_images/kaust/h5',
                        help="Path to directory with h5 HSI reflectances from Li dataset")
    parser.add_argument('-i', '--illums_csv', type=Path,
                        default=awb.illuminants_path / 'awb' / 'std.csv',
                        help="Path to csv file with illuminants dataset", )
    parser.add_argument('-p', '--params_json', type=Path,
                        default=None,
                        help=f'Path to json with parameters of simulate, if None default_paramters will be used={default_parameters}')
    parser.add_argument('-o', '--outdir', type=Path,
                        default='../../../images',
                        help="Path to the results of simulation")
    parser.add_argument('-n', '--number_process', type=int,
                        default=1,
                        help='Number of processes to simulate data')
    return parser.parse_args()


def load_sens_camera():
    path = str(Path(awb.cameras_path / 'iitp/canon600d.csv'))
    camera_sens = awb.SpectralSensitivies.from_csv(path, normalized=False, wavelength_key='wavelength',
                                                   channels_names=('red', 'green','blue'))
    return camera_sens

def read_h5(h5_path: Path):
    with h5py.File(h5_path, 'r') as fh:
        hsi_y = np.transpose(fh['img\\']).astype(np.float32)
        return hsi_y


@dataclass
class ReflectancesBasedImageSimulator:
    outdir: Path
    illuminants_path: Path
    dsize: Tuple[int, int] = 512 // 4  # resizing to 128 x 128 x 34
    recalc: bool = False
    illum_names: List[str] = None
    debug: bool = False

    def __post_init__(self):
        ##====================
        new_illum = pd.read_csv('./spectr.csv')
        ##====================
        # illuminants
        illum_wv, illum_y_dict = awb.spectral_data.read_csv('./spectr.csv')
        # print(illum_wv.shape)
        if self.illum_names is None:
            self.spectra = {name: awb.SpectralFunction(illum_y, illum_wv)
                            for name, illum_y in new_illum.items()}
        else:
            if not set(illum_y_dict.keys()).intersection(set(self.illum_names)):
                raise RuntimeError(
                    'Defined by user illumin_names are different from names of input illuminants csv file!')
            self.spectra = {name: awb.SpectralFunction(illum_y, illum_wv)
                            for name, illum_y in illum_y_dict.items() if name in self.illum_names}
        # camera sensitivities
        self.sensitivities = {
            'cam': load_sens_camera()
        }


    def hsi_from_h5(self, h5_path):
        hsi_y = read_h5(h5_path)

        assert hsi_y.shape[0] % self.dsize == 0
        assert hsi_y.shape[1] % self.dsize == 0

        bin_size = hsi_y.shape[0] // self.dsize
        hsi_y = hsi_y.reshape((self.dsize, bin_size,
                               self.dsize, bin_size,
                               hsi_y.shape[-1]))

        dhsi_y = np.mean(hsi_y, axis=(1, 3))
        hsi = awb.SpectralFunction(dhsi_y, wv_hsi, dtype=np.float32)
        return hsi

    def hsi2img(self, hsi, sens):
        img = awb.spectral_rendering.camera_render(hsi, sens)
        img_max = np.max(img)
        img /= img_max

        img = np.clip(img, 0, 1)
        return img, img_max

    def sim_img(self, hsi, sens, outpath):
        img, _ = self.hsi2img(hsi, sens)
        awb.imsave(outpath, img, unchanged=False,
                   out_dtype=np.uint16, gamma_correction=False)

    def __call__(self, h5_path):
        for sens_name in self.sensitivities.keys():
            (self.outdir / sens_name).mkdir(parents=True, exist_ok=True)

        hsi = self.hsi_from_h5(h5_path)
        for spectrum_name, illum in self.spectra.items():
            if spectrum_name[:14] == str(h5_path)[45:59]:
                hsi_illum = hsi * illum

                # render rgb images for each sensor
                for sens_name, sens in self.sensitivities.items():
                    outpath = self.outdir / sens_name / \
                            h5_path.with_suffix(f'.{spectrum_name[-1]}.png').name

                    if not outpath.is_file() or self.recalc:
                        self.sim_img(hsi_illum, sens, outpath)



if __name__ == '__main__':
    args = parse_args()
    args.outdir.mkdir(exist_ok=True, parents=True)

    # using awb.SpectralFunction default wv grid
    wv_illum_spectra = np.linspace(400, 730, num=80, endpoint=True)
    awb.SpectralFunction.lambdas = wv_illum_spectra
    # decreasing time and memory consuming
    awb.SpectralFunction = partial(awb.SpectralFunction, dtype=np.float32)

    h5_paths = list(args.h5_path.glob('*.h5'))

    if args.params_json is None:
        params = default_parameters
    else:
        with open(args.params_json, 'r') as fh:
            params = json.load(fh)

    sim = ReflectancesBasedImageSimulator(
        outdir=args.outdir,
        illuminants_path=args.illums_csv,
        **params
    )

    # for path in tqdm.tqdm(h5_paths):
    #     sim(path)

    with Pool(ncpus=args.number_process) as p:
        with tqdm.tqdm(total=len(h5_paths)) as pbar:
            for _ in enumerate(p.imap(sim, h5_paths)):
                pbar.update()

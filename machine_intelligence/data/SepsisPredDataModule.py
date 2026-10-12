import lightning as lit

import wandb
from omegaconf import OmegaConf
from pathlib import Path
from loguru import logger

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from torchsampler import ImbalancedDatasetSampler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from machine_intelligence.data.SepsisPredDataset import SepsisPredictionDataset
import torchvision as tv

from machine_intelligence.data.TransformsData import SampleAndHold, ZeroFill
from machine_intelligence.utilities.data_handling import load_parquet_table

# lit.seed_everything(42, workers=True)

class SepsisPredDataModule(lit.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.generator = torch.Generator().manual_seed(42)
        #
        # lit.seed_everything(42, workers=True)
        # torch.manual_seed(42)

        # self.transforms = ZeroFill()

        self.transforms = tv.transforms.Compose([
            # ZeroFill(),
            SampleAndHold(),
            #  LowPassButter()
            #  WaveletDenoising()
        ]
        )
        self.batch_size = cfg.params.batch_size_gpu
        self.n_workers: int = cfg.experiment.n_workers

        self.cfg = cfg
        # dataset
        self.train_set_path: Path = Path(cfg.data.paths.df_train)
        self.val_set_path: Path = Path(cfg.data.paths.df_val)
        self.test_set_path: Path = Path(cfg.data.paths.df_test)

        # the static ds doesn't need a split, since the information is retrieved via the patient id
        self.static_variables_path: Path = Path(cfg.data.paths.static)

        self.tracked_run: bool = cfg.experiment.tracked_run
        dataset_name: str = cfg.data.name
        debug_dataset: bool = cfg.data.debug

        self.save_hyperparameters()

        logger.debug(f"{debug_dataset = }")

        logger.debug(f"{self.train_set_path = }")
        logger.debug(f"{self.val_set_path = }")
        logger.debug(f"{self.test_set_path = }")

        self.df_train = load_parquet_table(
            parquet_path=self.train_set_path, verbose=True, backend="numpy_nullable"
        )
        self.df_val = load_parquet_table(
            parquet_path=self.val_set_path, verbose=True, backend="numpy_nullable"
        )
        self.df_test = load_parquet_table(
            parquet_path=self.test_set_path, verbose=True, backend="numpy_nullable"
        )

        self.df_static = load_parquet_table(
            parquet_path=self.static_variables_path,
            verbose=False,
            backend="numpy_nullable",
        )

        all_columns: list = self.df_train.columns.tolist()
        feature_subset: list = OmegaConf.to_object(
            self.cfg.data.considered_feature_subset_training
        )
        aux_columns: list = OmegaConf.to_object(self.cfg.data.auxillary_columns)
        ignored_features: list = OmegaConf.to_object(self.cfg.data.ignored_columns)
        target: str = self.cfg.data.target_column

        if len(feature_subset) == 0:
            considered_features = list(
                set(all_columns) - {target} - set(aux_columns) - set(ignored_features)
            )
        else:
            considered_features = list(
                set(feature_subset)
                - {target}
                - set(aux_columns)
                - set(ignored_features)
            )

        considered_features_static_cat = OmegaConf.to_object(
            self.cfg.data.considered_features_static_cat
        )
        considered_features_static_num = OmegaConf.to_object(
            self.cfg.data.considered_features_static_num
        )
        considered_features_static = [
            *considered_features_static_cat,
            *considered_features_static_num,
        ]

        logger.info(f"{all_columns = }")
        logger.info(f"{aux_columns = }")
        logger.info(f"{ignored_features = }")
        logger.info(f"{considered_features = }")
        logger.info(f"{considered_features_static = }")
        logger.info(f"{target = }")

        sample_index_column: str = self.cfg.data.sample_idx_column

        self.all_columns = all_columns
        self.aux_columns = aux_columns
        self.ignored_columns = ignored_features
        self.considered_features = considered_features
        self.considered_features_static = considered_features_static
        self.considered_features_static_cat = considered_features_static_cat
        self.considered_features_static_num = considered_features_static_num
        self.target = target
        self.sample_index_column = sample_index_column

        # saving some metadata
        if self.tracked_run:
            # Note: These features include the "sepsis_present", it only used as a target var
            wandb.config["columns_considered"] = considered_features

        # self.class_weights = self.calculate_class_weight()

        self.df_static_train_processed, self.df_static_val_processed, self.df_static_test_processed = self.prepare_static_data()

        self.class_weights = self.calculate_class_weight()

        class_counts_train = self.df_train.drop_duplicates(subset=self.sample_index_column)[self.target].value_counts()
        self.class_weight_wrs_train = 1 / class_counts_train

        class_counts_val = self.df_val.drop_duplicates(subset=self.sample_index_column)[self.target].value_counts()
        self.class_weight_wrs_val = 1 / class_counts_val

        df_train_sample_target = self.df_train.drop_duplicates(subset=sample_index_column)
        df_train_sample_target["class_weight"] = df_train_sample_target[target].map(
            lambda x: self.class_weights[int(x)])

        df_val_sample_target = self.df_val.drop_duplicates(subset=sample_index_column)
        df_val_sample_target["class_weight"] = df_val_sample_target[target].map(lambda x: self.class_weights[int(x)])

        self.sample_weights_train = df_train_sample_target["class_weight"].to_numpy()
        self.sample_weights_val = df_val_sample_target["class_weight"].to_numpy()
        # getting cohort information on the test set
        # self.df_static_test_processed

    def prepare_data(self):
        pass

    def setup(self, stage: str):

        if stage == "fit":
            self.train_dataset = SepsisPredictionDataset(
                dataframe_time_series=self.df_train,
                dataframe_static=self.df_static_train_processed,
                target=self.target,
                features=self.considered_features,
                features_static_cat=self.considered_features_static_cat,
                features_static_num=self.considered_features_static_num,
                sample_index_column_name=self.sample_index_column,
                auxillary_columns=self.aux_columns,
                transform=self.transforms,
            )
            self.val_dataset = SepsisPredictionDataset(
                dataframe_time_series=self.df_val,
                dataframe_static=self.df_static_val_processed,
                target=self.target,
                features=self.considered_features,
                features_static_cat=self.considered_features_static_cat,
                features_static_num=self.considered_features_static_num,
                sample_index_column_name=self.sample_index_column,
                auxillary_columns=self.aux_columns,
                transform=self.transforms,
            )
        if stage == "test":
            self.test_dataset = SepsisPredictionDataset(
                dataframe_time_series=self.df_test,
                dataframe_static=self.df_static_test_processed,
                target=self.target,
                features=self.considered_features,
                features_static_cat=self.considered_features_static_cat,
                features_static_num=self.considered_features_static_num,
                sample_index_column_name=self.sample_index_column,
                auxillary_columns=self.aux_columns,
                transform=self.transforms,
            )

    def train_dataloader(self):

        self.sampler_train = WeightedRandomSampler(weights=self.sample_weights_train,
                                                   num_samples=self.train_dataset.__len__(),
                                                   replacement=True,
                                                   generator=self.generator)

        train_data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            # shuffle=True,
            # pin_memory=True,
            num_workers=self.n_workers,
            # prefetch_factor=2,
            generator=self.generator,
            persistent_workers=True,
            # sampler=ImbalancedDatasetSampler(self.train_dataset)
            sampler=self.sampler_train
        )
        return train_data_loader

    def val_dataloader(self):

        self.sampler_val = WeightedRandomSampler(weights=self.sample_weights_val,
                                                 num_samples=self.val_dataset.__len__(),
                                                 replacement=True)

        val_data_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            # pin_memory=True,
            num_workers=self.n_workers,
            persistent_workers=True,
            # prefetch_factor=3,
            generator=self.generator,

            # sampler=self.sampler_val # don't oversample the validation set as this is would be overly optimistic
        )
        return val_data_loader

    def test_dataloader(self):
        test_data_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            # pin_memory=True,
            # num_workers=self.n_workers,
            num_workers=0,
            # persistent_workers=True,
            # prefetch_factor=3,
            # generator=self.generator,

        )

        return test_data_loader


    def calculate_class_weight(self):
        df_patients_and_sepsis_status = self.df_train[
            [
                "sepsis_sample",
                "sepsis_sample_shifted",
                "patientunitstayid",
                "sample_idx_6_h",
            ]
        ]

        label_classes = (
            df_patients_and_sepsis_status.drop_duplicates(
                ["sepsis_sample_shifted", "sample_idx_6_h"]
            )["sepsis_sample_shifted"]
            .value_counts()
            .to_dict()
        )
        total_number_samples: int = len(
            self.df_train[self.sample_index_column].unique()
        )
        total_number_classes: int = len(label_classes)

        class_weights: list = [
            total_number_samples / (total_number_classes * label_classes[key])
            for key in label_classes
        ]
        logger.info(f"Class weights: {class_weights}")

        return class_weights

    def prepare_static_data(self):
        # static data dtype mapping
        dtype_mapping = {
            "gender": "category",
            "aids": "boolean",
            "hepaticfailure": "boolean",
            "lymphoma": "boolean",
            "metastaticcancer": "boolean",
            "leukemia": "boolean",
            "immunosuppression": "boolean",
            "diabetes": "boolean",
            "admissionweight": "float",
            "admissionheight": "float",
            "age": "Int64",
        }

        # Apply the data types to the DataFrame
        for col, dtype in dtype_mapping.items():
            self.df_static[col] = self.df_static[col].astype(dtype)

        self.df_static["gender"] = self.df_static["gender"].map({"Female": 1, "Male": 0})

        df_static_train = self.df_static.loc[
            self.df_static["patientunitstayid"].isin(
                self.df_train["patientunitstayid"]
            ),
            ["patientunitstayid", *self.considered_features_static],
        ]

        df_static_val = self.df_static.loc[
            self.df_static["patientunitstayid"].isin(self.df_val["patientunitstayid"]),
            ["patientunitstayid", *self.considered_features_static],
        ]

        df_static_test = self.df_static.loc[
            self.df_static["patientunitstayid"].isin(self.df_test["patientunitstayid"]),
            ["patientunitstayid", *self.considered_features_static],
        ]

        # scaling and one-hot encoding the static variables based on the results of rus
        oh_encoder = OneHotEncoder(
            handle_unknown="ignore", drop=None, sparse_output=False
        )
        oh_encoder.fit(df_static_train[self.considered_features_static_cat])

        static_ohe_train = oh_encoder.transform(
            df_static_train[self.considered_features_static_cat]
        )
        static_ohe_val = oh_encoder.transform(
            df_static_val[self.considered_features_static_cat]
        )
        static_ohe_test = oh_encoder.transform(
            df_static_test[self.considered_features_static_cat]
        )

        # scaler = StandardScaler()
        scaler = MinMaxScaler()
        scaler.fit(X=df_static_train[self.considered_features_static_num])
        static_scaled_train = scaler.transform(
            df_static_train[self.considered_features_static_num]
        )
        static_scaled_val = scaler.transform(
            df_static_val[self.considered_features_static_num]
        )
        static_scaled_test = scaler.transform(
            df_static_test[self.considered_features_static_num]
        )

        # df_static_train_processed = pd.DataFrame(
        #     data=static_ohe_train, columns=oh_encoder.get_feature_names_out()
        # )
        df_static_train_processed = df_static_train.copy()
        # df_static_train_processed["patientunitstayid"] = df_static_train["patientunitstayid"].to_numpy()
        df_static_train_processed[scaler.get_feature_names_out()] = static_scaled_train

        df_static_train_processed = df_static_train_processed
        # df_static_val_processed = pd.DataFrame(
        #     data=static_ohe_val, columns=oh_encoder.get_feature_names_out()
        # )
        df_static_val_processed = df_static_val.copy()
        # df_static_val_processed["patientunitstayid"] = df_static_val["patientunitstayid"].to_numpy()
        df_static_val_processed[scaler.get_feature_names_out()] = static_scaled_val

        df_static_val_processed = df_static_val_processed
        # df_static_test_processed = pd.DataFrame(
        #     data=static_ohe_test, columns=oh_encoder.get_feature_names_out()
        # )

        df_static_test_processed = df_static_test.copy()
        # df_static_test_processed["patientunitstayid"] = df_static_test["patientunitstayid"].to_numpy()
        df_static_test_processed[scaler.get_feature_names_out()] = static_scaled_test

        df_static_test_processed = df_static_test_processed

        return (
            df_static_train_processed,
            df_static_val_processed,
            df_static_test_processed,
        )

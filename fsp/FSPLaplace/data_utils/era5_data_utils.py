import jax
import math

import numpy as np
import xarray as xr
import jax.numpy as jnp


r_earth = 1 # 6371.0  # km
C_earth = 2. * jnp.pi * r_earth  # km

N_long_total = 2 * 2 * 2 * 2 * 2 * 3 * 3 * 5  # = 1440
#             |_____| |_____________________|
#              = 4           = 360
 
N_lat_total = 2 * 2 * 2 * 2 * 3 * 3 * 5 + 1  # = 721
#            |_____| |_________________|
#              = 4         = 180



def gcs_to_cartesian(
    longitude, 
    latitude, 
    r=r_earth,
    stats=None
):
    x = r * jnp.cos(latitude) * jnp.cos(longitude)
    y = r * jnp.cos(latitude) * jnp.sin(longitude)
    z = r * jnp.sin(latitude)
    
    return jnp.stack([x, y, z], axis=-1)


class ERA5Dataset:
    
    def __init__(
        self, 
        ds_path,
        t_idcs=slice(0,7*24), # 7 days
        step_long=4,
        step_lat=4
    ):  
        assert N_long_total % step_long == 0, "N_long_total is not divisible by step_long"
        assert (N_lat_total - 1) % step_lat == 0, "N_lat_total - 1 is not divisible by step_lat"

        # Attributes
        self.dataset = xr.open_dataset(ds_path)
        self.feature_stats = {"x": {}, "y": {}, "z": {}, "t": {}}

        # Subsample longitude and latitude
        assert N_long_total == len(self.dataset["longitude"])
        #assert N_lat_total == len(self.dataset["latitude"])
        longitude = self.dataset["longitude"][::step_long] * (jnp.pi / 180.0) # rad
        latitude = self.dataset["latitude"][::step_lat][::-1] * (jnp.pi / 180.0)  # rad - 401

        # Subsample time
        ts = (self.dataset["time"][t_idcs] - self.dataset["time"][0]) / np.timedelta64(1, 'h') #  to hours

        # Load the entire dataset to memory
        with jax.default_device(jax.devices("cpu")[0]):
            # Build the spacial grid
            longitude = jnp.array(longitude.values)
            latitude = jnp.array(latitude.values)
            
            # Build the temporal grid
            ts = jnp.array(ts.values) # 0-1 normalization
            if ts.shape[0] > 1:
                self.feature_stats["t"] = {"mean": ts.mean(), "std": ts.std()}
            else:
                self.feature_stats["t"] = {"mean": ts.mean(), "std": 1}
            # Build the feature grid
            self.X = (longitude, latitude, ts)
            # Build the labels
            t2m = self.dataset["t2m"][t_idcs, ::step_lat, ::step_long] # time, latitude, longitude
            # Reverse latitude + transpose + convert to °C
            self.y = jnp.flip(jnp.array(t2m.values).T, axis=1) - 273.15 # °C - long, lat, time
            print(f"y: {self.y.shape} - Memory: {self.y.nbytes / 1e6} MB")
            # ##### 
            # import matplotlib.pyplot as plt
            # def autocorr(x):
            #     x = x - x.mean()
            #     result = np.correlate(x, x, mode='full')
            #     return result[result.size // 2:]
            # # for i in range(len(bins)):
            # print(self.dataset["t2m"])
            # f = self.dataset["t2m"].isel(latitude=90*4-9*4, longitude=77*4).plot()
            # plt.show()
            # x = self.dataset["t2m"].isel(latitude=90*4-9*4, longitude=77*4).values.reshape(-1)
            # acorr = autocorr(x)
            # plt.plot(np.arange(0, acorr.shape[0]), acorr)
            # plt.show()
            # fft = np.fft.fft(acorr)
            # frequency_axis = np.fft.fftfreq(acorr.shape[0], d=1.0)
            # norm_amplitude = np.abs(fft) /(acorr.shape[0] / 2) 
            # plt.plot(frequency_axis, norm_amplitude)
            # plt.xlabel('Frequency[Hz]')
            # plt.ylabel('Amplitude')
            # plt.title('Spectrum')
            # max_idx = np.argmax(norm_amplitude[1:]) + 1
            # print(f"Max frequency: {frequency_axis[max_idx]} Hz")
            # plt.show()
            # exit()
            # #######
            # Compute grid statistics for GP kernels
            self.time_std = t_idcs.step * self.feature_stats["t"]["std"]

        # Close the dataset
        self.dataset.close()

        # Update feature min and max
        normalized_t = (ts - self.feature_stats["t"]["mean"]) / self.feature_stats["t"]["std"]
        self.x_min = jnp.concatenate([self.X[0].min().reshape(1), self.X[1].min().reshape(1), normalized_t.min().reshape(1)], axis=0)
        self.x_max = jnp.concatenate([self.X[0].max().reshape(1), self.X[1].max().reshape(1), normalized_t.max().reshape(1)], axis=0)


    def __len__(
        self
    ):
        return np.prod(self.y.shape)


    def __getitem__(
        self, 
        idx
    ):
        # Compute index : (longitude, latitude, time)
        longitude_idx, latitude_idx, time_idx = jnp.unravel_index(idx, self.y.shape)

        # Get the normalized feature vector
        longitude, latitude, ts = self.X[0][longitude_idx], self.X[1][latitude_idx], self.X[2][time_idx]
        ts = (ts - self.feature_stats["t"]["mean"]) / self.feature_stats["t"]["std"]
        x = jnp.concatenate([longitude.reshape(-1,1), latitude.reshape(-1,1), ts.reshape(-1,1)], axis=-1)
        #xyz = gcs_to_cartesian(longitude, latitude, stats=self.feature_stats)
        #x = jnp.concatenate([xyz.reshape(-1,3), ts.reshape(-1,1)], axis=-1)

        # Get the label
        y = self.y[longitude_idx, latitude_idx, time_idx].reshape(-1, 1)

        return x, y


class ERA5DataLoader:
    """
    Object that samples from a dataset.
    """
    def __init__(
        self, 
        key, 
        dataset, 
        batch_size,
        dataset_idx, 
        shuffle=False, 
        normalize_labels=True
    ):
        """
        Build DataLoader object.

        params:
        - key (jax.random.PRNGKey): random key.
        - dataset (Dataset): dataset to sample from.
        - batch_size (int): batch size.
        - shuffle (bool): shuffle the dataset at each epoch.
        - replacement (bool): sample dataset with replacement.
        """
        self.key = key
        self.dataset = dataset
        self.batch_size = batch_size
        self.replacement = False
        self.shuffle = shuffle
        self.dataset_idx = dataset_idx
        self.normalize_labels = normalize_labels
        self.feature_dim = 3

        if self.normalize_labels:
            longitude_idx, latitude_idx, time_idx = jnp.unravel_index(dataset_idx, self.dataset.y.shape)
            y = self.dataset.y[longitude_idx, latitude_idx, time_idx].reshape(-1, 1)
            self.label_stats = {"mean": y.mean(), "std": y.std()}

        # Shuffle dataset
        self.idxs = jax.random.choice(
            self.key, 
            self.dataset_idx, 
            shape=(self.dataset_idx.shape[0],), 
            replace=False
        )

    
    def __len__(
        self
    ):
        """
        Length of the dataloader i.e. number of batches.

        returns:
        - len (int): number of batches.
        """
        return math.ceil(self.dataset_idx.shape[0] / self.batch_size)
    
    
    def __iter__(
        self
    ):
        """
        Build iterator.

        returns:
        - self (DataLoader): iterator.
        """
        # Reset iterator
        self.it = 0
        
        # Shuffle dataset
        if self.shuffle:
            self.key, subkey = jax.random.split(self.key)
            self.idxs = jax.random.choice(
                subkey, 
                self.dataset_idx, 
                shape=(self.dataset_idx.shape[0],), 
                replace=False
            )
        
        return self
    

    def __next__(
        self
    ):
        """
        Sample a batch from the dataset.

        returns:
        - out (jnp.array): batch of data.
        """
        # If end of array is reached
        if self.it >= len(self):    
            raise StopIteration
        
        # Select start and end indices
        with jax.default_device(jax.devices()[0]):
            start = self.it * self.batch_size
            if (self.it+1) > len(self): 
                out = self.dataset[self.idxs[start:]]
            else:
                end = (self.it+1) * self.batch_size
                out = self.dataset[self.idxs[start:end]]

            # Normalize the data
            if self.normalize_labels:
                # Normalize the label
                x, y = out 
                y = (y - self.label_stats["mean"]) / self.label_stats["std"]
                out = (x, y)

        # Update iterator
        self.it += 1

        return out
    

    def set_replacement_mode(
        self, 
        replacement
    ):
        pass




class ERA5PatchDataLoader:

    def __init__(
        self, 
        key, 
        dataset,
        patch_shape={"time": 1, "latitude": 10, "longitude": 10},
        offset="random",  # "none" or "random"
    ):  
        # Attributes
        self.key = key
        self.feature_dim = 3
        self.offset_type = offset
        self.replacement = False
        self.patch_shape = patch_shape
        self.dataset = dataset
        self.X = dataset.X
        self.y = dataset.y
        self.feature_stats = dataset.feature_stats
        self.label_stats = {"mean": self.y.mean(), "std": self.y.std()}

        # assert self.X[0].shape[0] // self.patch_shape["longitude"] != 0, "Patch size is larger than the longitude dimension"
        # assert self.X[1].shape[0] // self.patch_shape["latitude"] != 0, "Patch size is larger than the latitude dimension"
        # assert self.X[2].shape[0] // self.patch_shape["time"] != 0, "Patch size is larger than the time dimension"
        
        # Build iterator
        self.it = 0

        print("X:", self.X[0].shape, self.X[1].shape, self.X[2].shape)
        print(f"y: {self.y.shape} - Memory: {self.y.nbytes / 1e6} MB")



    def __len__(self):
        """
        Number of data points.

        returns:
        - len (int): number of data points.
        """
        return math.prod(s // self.patch_shape[key] for s, key in zip(self.y.shape, ["longitude", "latitude", "time"]))
    

    def __iter__(
        self
    ):
        """
        Build iterator.

        returns:
        - self (DataLoader): iterator.
        """
        # Reset iterator
        self.it = 0
        
        return self
    

    def __next__(
        self
    ):
        """
        Sample a batch from the dataset.

        returns:
        - out (jnp.array): batch of data.
        """
        # If end of array is reached
        if self.it >= len(self):   
            raise StopIteration
        
        idx = self.it
        with jax.default_device(jax.devices()[0]):
            # Compute index : (longitude, latitude, time)
            coord_idcs = {}
            for coord_name, coord_value in self.patch_shape.items():
                if coord_name == "longitude":
                    coordinate = self.X[0]
                elif coord_name == "latitude":
                    coordinate = self.X[1]
                elif coord_name == "time":
                    coordinate = self.X[2]

                dim_size = coordinate.shape[0]
                num_entries = dim_size // coord_value
                idx, current = divmod(idx, num_entries)

                # Offset
                if self.offset_type == "random":
                    self.key, subkey = jax.random.split(self.key)
                    #maxval = dim_size if coord_name != "time" else dim_size - (current + 1) * coord_value # time cannot be offset
                    maxval = dim_size - (current + 1) * coord_value # time cannot be offset
                    offset = jax.random.randint(subkey, minval=0, maxval=maxval, shape=(1,)).item() 
                else:
                    offset = 0

                # Compute the start and end indices
                start = (current * coord_value + offset) % dim_size
                end = ((current + 1) * coord_value + offset) % dim_size

                if end == 0:  # else the wrap around is incorrect
                    end = dim_size

                start_val = coordinate[start]
                end_val = coordinate[end]

                bigger_equal_start = coordinate >= start_val
                smaller_than_end = coordinate < end_val

                if start < end:
                    coord_idcs[coord_name] = slice(start, end)
                else:  # wrap-around case
                    coord_idcs[coord_name] = bigger_equal_start | smaller_than_end

            # Get the feature vector
            longitudes = self.X[0][coord_idcs["longitude"]].reshape(-1, 1)
            latitudes = self.X[1][coord_idcs["latitude"]].reshape(-1, 1)
            times = self.X[2][coord_idcs["time"]].reshape(-1, 1)
            times = (times - self.feature_stats["t"]["mean"]) / self.feature_stats["t"]["std"]
            x = jax.vmap(lambda lon:
                jax.vmap(lambda lat: 
                    jax.vmap(lambda t: jnp.concatenate([lon.reshape(-1, 1), lat.reshape(-1, 1), t.reshape(-1, 1)]))(times)
                )(latitudes)
            )(longitudes).reshape(-1, 3)
            # x = jax.vmap(lambda lon:
            #     jax.vmap(lambda lat: 
            #         jax.vmap(lambda t: jnp.concatenate([gcs_to_cartesian(lon, lat, stats=self.feature_stats).reshape(-1, 1), t.reshape(-1, 1)]))(times)
            #     )(latitudes)
            # )(longitudes).reshape(-1, 4)

            # Get the label
            y = self.y[coord_idcs["longitude"], coord_idcs["latitude"], coord_idcs["time"]].reshape(-1, 1)
            y = (y - self.label_stats["mean"]) / self.label_stats["std"]

        # Update iterator
        self.it += 1

        return x, y
    
    def set_replacement_mode(
        self, 
        replacement
    ):
        pass
    

if __name__ == "__main__":

    import time 
    start_time = time.time()
    dataset = ERA5Dataset(ds_path="/Users/tristancinquin/Downloads/era5_t2m.nc", t_idcs=slice(0,30*24,1), step_long=20, step_lat=20)
    print("Min X:", dataset.x_min)
    print("Max X:", dataset.x_max)
    print("Time to load dataset:", time.time() - start_time)
    print("Dataset length:", len(dataset))

    # start_time = time.time()
    # key = jax.random.PRNGKey(0)
    # dataset_idx = jnp.arange(len(dataset))
    # batch_size = 100
    # dataloader = ERA5DataLoader(key, dataset, batch_size, dataset_idx, shuffle=False, normalize_labels=True)
    # print("Time to load dataloader:", time.time() - start_time)
    # print("Dataset length:", len(dataloader))

    # start_time = time.time()
    # for i, (x, y) in enumerate(dataloader):
    #     print(f"{i} samples", x.shape, y.shape)
    #     break 
    # print("Time to iterate over dataloader:", time.time() - start_time)

    # start_time = time.time()
    # dataloader = ERA5PatchDataLoader(key, dataset, patch_shape={"time": 1, "latitude": 10, "longitude": 10}, offset="random")
    # print("Time to load ERA5PatchDataloader:", time.time() - start_time)
    # print("Dataset length:", len(dataloader))

    # start_time = time.time()
    # for i, (x, y) in enumerate(dataloader):
    #     if i % 100 == 0:
    #         print(f"{i} samples", x.shape, y.shape)
    # print("Time to iterate over ERA5PatchDataloader:", time.time() - start_time)

    print("Plot")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})    


    time = 0
    X = dataset.X
    t = dataset.y[:,:,time] # long, lat, time

    xyzs = jax.vmap(lambda lon:
        jax.vmap(lambda lat: gcs_to_cartesian(lon, lat, stats=dataset.feature_stats))(X[1]) # long, lat
    )(X[0])

    print("x mean", xyzs[:,:,0].mean(), xyzs[:,:,0].std())
    print("y mean", xyzs[:,:,1].mean(), xyzs[:,:,1].std())
    print("z mean", xyzs[:,:,2].mean(), xyzs[:,:,2].std())
    
    # Plot the surface.
    print("xyzs:", xyzs.shape)
    print("t.shape", t.shape)
    norm=Normalize(vmin=t.min(), vmax=t.max())
    my_col = cm.coolwarm(norm(t))
    print("my_col.shape", my_col.shape)
    print(t.min(), t.max())
    
    surf = ax.plot_surface(
        jnp.flip(jnp.concatenate([xyzs[:, :, 0], xyzs[0, :, 0].reshape(1, -1)], axis=0), 0), 
        jnp.flip(jnp.concatenate([xyzs[:, :, 1], xyzs[0, :, 1].reshape(1, -1)], axis=0), 0),
        jnp.flip(jnp.concatenate([xyzs[:, :, 2], xyzs[0, :, 2].reshape(1, -1)], axis=0), 0),
        facecolors=my_col,
        vmin=t.min(),
        vmax=t.max(),
        cmap=cm.coolwarm,
        linewidth=0, 
        antialiased=False
    )

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5, ax=ax)

    plt.show()

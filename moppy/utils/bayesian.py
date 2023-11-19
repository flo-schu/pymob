import xarray as xr

def cluster_chains(posterior):
    assert isinstance(posterior, xr.DataArray)
    chain_means = posterior.mean(dim="draw")
    chain_sd = posterior.std(dim="draw")
    

    global cluster_id
    cluster_id = 1
    cluster = [cluster_id] * len(posterior.chain)
    unclustered_chains = posterior.chain.values

    def recurse_clusters(unclustered_chains):
        global cluster_id
        compare = unclustered_chains[0]
        new_cluster = []
        for i in unclustered_chains[1:]:
            a = chain_means.sel(chain=compare) + chain_sd.sel(chain=compare) > chain_means.sel(chain=i)
            b = chain_means.sel(chain=compare) - chain_sd.sel(chain=compare) < chain_means.sel(chain=i)

            if not all(a * b):
                cluster[i] = cluster_id + 1
                new_cluster.append(i)

        cluster_id += 1
        if len(new_cluster) == 0:
            return

        recurse_clusters(new_cluster)
    
    recurse_clusters(unclustered_chains)

    return cluster
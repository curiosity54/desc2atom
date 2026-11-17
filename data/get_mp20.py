import ase
import polars as pl

# data from https://huggingface.co/datasets/OMatG/Alex-MP-20/blob/main/test.parquet
df = pl.read_parquet("/Users/jigyasa/scratch/reconstruct-descriptor/data/MP20_test.parquet")
df = df.to_pandas()

def row_to_atoms(row) :
    Z = np.asarray(row["atomic_numbers"], dtype=int)
    # print(Z)
    # print( len(row["positions"]), np.vstack(row["positions"]).shape, len(row["cell"]), np.vstack(row["cell"]).shape)
    pos = np.asarray(np.vstack(row["positions"]), dtype=float)
    cell = np.asarray(np.vstack(row["cell"]), dtype=float).reshape(3, 3)
    return ase.Atoms(numbers=Z, positions=pos, cell=cell, pbc=True)
    
frames = [row_to_atoms(df.iloc[i]) for i in range(0,len(df),10)]


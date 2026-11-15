"""Candidate annotation checks; consistency is not independent TCGA validation."""
import pandas as pd
from config import DATA_DIR


def load_crosswalk() -> pd.DataFrame:
    """Check the saved crosswalk against its local third-party source."""
    cross = pd.read_csv(DATA_DIR / "ONCOSENTRY_gene_crosswalk.csv", dtype=str).fillna("")
    source = pd.read_csv(DATA_DIR / "third_party_gene_mapping.csv", dtype=str).fillna("")
    expected = [f"gene_{i}" for i in range(20531)]
    if cross.uci_feature.tolist() != expected or source.uci_name.tolist() != expected:
        raise ValueError("Crosswalk must preserve all 20,531 original feature identifiers in order.")
    for column in ["uci_feature", "output_column", "entrez_id"]:
        if cross[column].duplicated().any() or cross[column].eq("").any():
            raise ValueError(f"Missing or duplicated mapping identifier: {column}")
    symbols = source.symbol.replace("?", "")
    if not (cross.gene_symbol.eq(symbols).all()
            and cross.entrez_id.eq(source.entrez_id).all()
            and cross.original_tcga_gene_id.eq(source.gene_id).all()):
        raise ValueError("Crosswalk metadata disagrees with its saved source.")
    for row in cross.itertuples():
        symbol = row.gene_symbol or f"ENTREZ_{row.entrez_id}"
        status = "third_party_symbol_not_independently_verified" if row.gene_symbol else "symbol_unresolved"
        if row.output_column != f"{symbol}__ENTREZ{row.entrez_id}" or row.mapping_status != status:
            raise ValueError(f"Inconsistent crosswalk row: {row.uci_feature}")
    if cross.gene_symbol.eq("").sum() != 29:
        raise ValueError("Expected 29 unresolved candidate symbols.")
    return cross


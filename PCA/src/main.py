import logging
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)  # Ensures logs appear in Docker logs
    ]
)
logger = logging.getLogger("data_pipeline")

class DataPipeline:
    def __init__(self, input_path):
        self.input_path = input_path
        self.output_dir = "data/outputs"
        os.makedirs(self.output_dir,exist_ok=True)
        logger.info(f"Initializing PCA Run: {input_path}")

    def extract(self):
        logger.info(f"Reading CSV from {self.input_path}")
        df = pd.read_csv(self.input_path)
        df = df.select_dtypes(include=['number'])
        # drop ROI number and other things
        cols_to_drop = ['roi_number','BoundingBox_xwidth','BoundingBox_ywidth']
        features_df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        return features_df
    
    def remove_outliers(self, data):
        logger.info("Detecting and removing outliers...")
        # n_neighbors=20 is standard; contamination is the % of data you expect is 'garbage'
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.02)
        good_data_mask = lof.fit_predict(data) == 1
        
        outlier_count = (~good_data_mask).sum()
        logger.info(f"Removed {outlier_count} outliers.")

        return data[good_data_mask]

    def perform_pca(self, data):
        logger.info("Scaling data and performing PCA...")
        
        scaled_data = StandardScaler().fit_transform(data)

        cleaned_scaled_data = self.remove_outliers(scaled_data)
        
        pca = PCA(n_components=3)
        principal_components = pca.fit_transform(cleaned_scaled_data)

        loadings = pd.DataFrame(
            pca.components_.T * np.sqrt(pca.explained_variance_), 
            columns=['PC1', 'PC2', 'PC3'], 
            index=data.columns
        )
        
        # Sort by PC1 to see which columns drive the most variance
        logger.info("Top features contributing to PC1:\n" + str(loadings['PC1'].sort_values(ascending=False).head(5)) )
        
        pca_df = pd.DataFrame(data=principal_components, 
                              columns=['PC1', 'PC2', 'PC3'])
        
        logger.info(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        
        return pca_df

    def save_graphs(self, pca_df):
        logger.info("Generating plot...")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], c='skyblue', s=60)
        ax.set_title('3D PCA Projection')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')

        output_path = f"{self.output_dir}/pca_result_{datetime.now().strftime('%d%m%Y_%H%M%S')}.png"
        plt.savefig(output_path)
        logger.info(f"Plot saved to {output_path}")

        return True

    def run(self):
        try:
            raw_data = self.extract()
            pca_results = self.perform_pca(raw_data)
            self.save_graphs(pca_results)
            logger.info("Pipeline completed successfully! ✅")
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            sys.exit(1)

if __name__ == "__main__":
    csv_path = os.getenv("INPUT_CSV", "data/input.csv")
    pipeline = DataPipeline(csv_path)
    pipeline.run()
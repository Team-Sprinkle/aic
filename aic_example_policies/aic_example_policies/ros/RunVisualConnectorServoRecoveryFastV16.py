"""Validated-codec sharing on the bounded SC recovery /8mm/s controller."""
from .RunVisualConnectorServoRecoveryV15 import RunVisualConnectorServoRecoveryV15
from shared_grasp_training_jpeg import SharedGraspTrainingJPEG,SharedJPEGEstimator


class RunVisualConnectorServoRecoveryFastV16(RunVisualConnectorServoRecoveryV15):
    def _warm_grasp(self):
        super()._warm_grasp()
        self.grasp_jpeg_cache=SharedGraspTrainingJPEG()
        self.grasp=SharedJPEGEstimator(self.grasp,self.grasp_jpeg_cache,begin=False)
        for family,model in list(self.connector_models.items()):
            if isinstance(model,SharedJPEGEstimator):model=model.estimator
            self.connector_models[family]=SharedJPEGEstimator(model,self.grasp_jpeg_cache,begin=True)
        self._trace(event='shared_native_grasp_jpeg_enabled',quality=86,codec_passes_per_triplet=1,
            model_weights_changed=False,geometry_input=False)

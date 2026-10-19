"""V26 control with the already validated exact shared-JPEG acquisition path."""
from .RunVisualConnectorServoRecoveryV26 import RunVisualConnectorServoRecoveryV26
from shared_grasp_training_jpeg import SharedGraspTrainingJPEG, SharedJPEGEstimator


class RunVisualConnectorServoRecoveryFastV27(RunVisualConnectorServoRecoveryV26):
    def _warm_grasp(self):
        super()._warm_grasp()
        self.grasp_jpeg_cache=SharedGraspTrainingJPEG()
        self.grasp=SharedJPEGEstimator(self.grasp,self.grasp_jpeg_cache,begin=False)
        for family,model in list(self.connector_models.items()):
            if isinstance(model,SharedJPEGEstimator):
                model=model.estimator
            self.connector_models[family]=SharedJPEGEstimator(model,self.grasp_jpeg_cache,begin=True)
        self._trace(event='shared_native_grasp_jpeg_enabled',quality=86,
                    codec_passes_per_triplet=1,model_weights_changed=False,geometry_input=False)

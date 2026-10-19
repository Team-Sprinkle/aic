"""Same image-based controller and models, with bit-exact shared grasp JPEG86."""
from .RunVisualConnectorServoDiagnosticV4 import RunVisualConnectorServoDiagnosticV4
from shared_grasp_training_jpeg import SharedGraspTrainingJPEG,SharedJPEGEstimator


class RunVisualConnectorServoFastV9(RunVisualConnectorServoDiagnosticV4):
    def _warm_grasp(self):
        super()._warm_grasp()
        self.grasp_jpeg_cache=SharedGraspTrainingJPEG()
        self.grasp=SharedJPEGEstimator(self.grasp,self.grasp_jpeg_cache,begin=False)
        for family,model in list(self.connector_models.items()):
            if isinstance(model,SharedJPEGEstimator):model=model.estimator
            self.connector_models[family]=SharedJPEGEstimator(model,self.grasp_jpeg_cache,begin=True)
        self._trace(event='shared_native_grasp_jpeg_enabled',quality=86,codec_passes_per_triplet=1,
            new_observation_scope='each direct connector-pose prediction',model_weights_changed=False,geometry_input=False)

from pathlib import Path
import yaml
import inspect

class BasicPathConfig:
    _PATH_FIELDS = {
        "MNN_root",
        "project_root",
        "onnx_data_root",
        "mnn_data_root",
        "backup_folder",
        "LLM_response_folder",
        "op_pymodel_foler_path",
    }

    def __init__(self):
        """
            Under this project file, create folders:
                1. backop_ops: copy the original MNN operator
                2. source: put the dataset here

            parameters:
                llm        : "gpt-5","claude-sonnet-4.5","gemini-2.5-flash","llama-3.1-405b-instruct"
                response_mod : "gen","read"
                api_source : ""openrouter","aliyun"
        """
        # -------- path fields --------
        self.MNN_root: Path | None = None
        self.project_root: Path | None = None
        self.onnx_data_root: Path | None = None
        self.mnn_data_root: Path | None = None
        self.backup_folder: Path | None = None
        self.LLM_response_folder: Path | None = None
        self.op_pymodel_foler_path: Path | None = None

        # -------- non-path --------
        self.cfg_LLM: str | None = None
        self.api_source: str | None = None
        self.response_mod: str | None = None

    def update_from_yaml(self, yaml_path: str | Path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        for k, v in config.items():
            if not hasattr(self, k):
                continue

            if k in self._PATH_FIELDS:
                setattr(self, k, Path(v) if v else None)
            else:
                setattr(self, k, v)

class OpConfig(BasicPathConfig):
    def __init__(
        self,
        op_name: str | None = None,
        op_type: str | None = None,
        op_ctg: str | None = None,
    ):
        super().__init__()

        self.op_name = op_name
        self.op_type = op_type
        self.op_ctg = op_ctg

        if self.op_name:
            self._init_op_paths()

    # ---------- runtime update ----------
    def update_op(
        self,
        op_name: str | None = None,
        op_type: str | None = None,
        op_ctg: str | None = None,
        LLM: str | None = None,
    ):
        if op_name is not None:
            self.op_name = op_name
        if op_type is not None:
            self.op_type = op_type
        if op_ctg is not None:
            self.op_ctg = op_ctg

        if LLM:
            self.cfg_LLM = LLM

        self._init_op_paths()

    # ---------- op-specific paths ----------
    def _init_op_paths(self):
        assert self.op_name and self.op_type and self.op_ctg, \
            "op_name / op_type / op_ctg must be set"

        self.op_pymodel_path = (
            self.op_pymodel_foler_path / self.op_type / self.op_ctg/ f"{self.op_name}.py"
        )
        self.op_onnx_path = (
            self.onnx_data_root / self.op_type / self.op_ctg / f"{self.op_name}.onnx"
        )
        self.op_mnn_path = (
            self.mnn_data_root / self.cfg_LLM / self.op_type / self.op_ctg / f"{self.op_name}.mnn"
        )

        self.response_file = (
            self.LLM_response_folder / self.cfg_LLM / self.op_type / self.op_ctg / f"{self.op_name}.txt"
        )

        if self.op_type == "Atomic" and self.op_ctg == "convolution":
            self.register_op_folder = self.MNN_root / "source/backend/cpu/compute"
        else:
            self.register_op_folder = {
                "Atomic": self.MNN_root / "source/backend/cpu",
                "Combiner": self.MNN_root / "tools/converter/source/optimizer/onnxextra",
                "Geometry": self.MNN_root / "source/geometry",
            }[self.op_type]

if __name__ == "__main__":
    cfg =OpConfig("1","2","3")
    cfg.from_yaml(Path("scripts/eva_config.yaml"))

    print(cfg)
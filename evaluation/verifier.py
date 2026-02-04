"""
UMS Verifier
程序化验证器，确保输出符合 UMS schema 并检测各类错误

验证规则（v0.2）：
1. schema 校验：合法 JSON 且满足 JSON Schema
2. missing-field policy：不允许省略顶层 key
3. enum/type 校验：laterality/study_view/state/unit 等必须在允许集合中
4. answerability 一致性：若 answerability.<field>=false，则该字段必须为 null
5. geometry consistency：bbox/mask 坐标合法
6. measurement unit：单位合法
7. measurement range：数值在合理区间
"""

import json
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False


class FailureType(Enum):
    """失败类型枚举"""
    NONE = "none"
    SCHEMA_VIOLATION = "schema_violation"
    MISSING_FIELD = "missing_field"
    INVALID_JSON = "invalid_json"
    ENUM_ERROR = "enum_error"
    ANSWERABILITY_INCONSISTENCY = "answerability_inconsistency"
    GEOMETRY_MISMATCH = "geometry_mismatch"
    UNIT_ERROR = "unit_error"
    RANGE_ERROR = "range_error"
    METADATA_MISSING = "metadata_missing"
    LATERALITY_ERROR = "laterality_error"
    VIEW_ERROR = "view_error"
    AMBIGUITY = "ambiguity"
    LEAKAGE_SUSPECTED = "leakage_suspected"


class Role(Enum):
    """训练角色枚举"""
    POSITIVE = "positive"       # pass 且字段可答 → L_tok + L_rank + L_vdep
    ABSTAIN = "abstain"         # pass 但字段不可答 → 只做 L_tok (目标为 null)
    NEGATIVE_ONLY = "negative-only"  # 内容错但格式对 → 只作为 hard negative
    DROP = "drop"               # schema_violation/leakage → 丢弃


@dataclass
class VerifierResult:
    """验证结果"""
    passed: bool
    failure_type: FailureType
    failure_message: str
    confidence: float
    role: Role
    details: Dict[str, Any]


class UMSVerifier:
    """
    UMS 程序化验证器

    输出：pass/fail + failure_type + confidence + role
    """

    # 合法的枚举值
    VALID_MODALITIES = {"CT", "CXR", "MRI", "US", "Fundus", "Dermoscopy", "Pathology", "Other"}
    VALID_LATERALITIES = {"left", "right", "bilateral", None}
    VALID_STUDY_VIEWS = {"AP", "PA", "LAT", None}
    VALID_FINDING_STATES = {"present", "absent", "uncertain", None}
    VALID_MEASUREMENT_STATES = {"measured", "uncertain", None}
    VALID_UNITS = {"mm", "cm", "mm2", "cm2", "mm3", "cm3", "ml", "L", None}
    VALID_SPACING_SOURCES = {"DICOM", "dataset_metadata", "unknown", None}

    # 必需的顶层字段
    REQUIRED_FIELDS = {
        "modality", "anatomy", "findings", "laterality", "study_view",
        "geometry", "measurements", "answerability", "uncertainty",
        "provenance", "verifier"
    }

    # 测量值的合理范围（示例）
    MEASUREMENT_RANGES = {
        "diameter": (0.1, 500),      # mm
        "area": (0.01, 10000),       # mm2
        "volume": (0.001, 100000),   # mm3
    }

    def __init__(
        self,
        schema_path: Optional[str] = None,
        strict_mode: bool = False,
    ):
        """
        Args:
            schema_path: JSON Schema 文件路径
            strict_mode: 严格模式（更多检查）
        """
        self.strict_mode = strict_mode
        self.schema = None

        if schema_path and JSONSCHEMA_AVAILABLE:
            with open(schema_path, 'r', encoding='utf-8') as f:
                self.schema = json.load(f)

    def verify(self, ums_json: Dict[str, Any]) -> VerifierResult:
        """
        验证 UMS JSON

        Args:
            ums_json: UMS JSON 字典

        Returns:
            VerifierResult
        """
        details = {}

        # 1. JSON Schema 校验
        if self.schema and JSONSCHEMA_AVAILABLE:
            try:
                jsonschema.validate(ums_json, self.schema)
            except jsonschema.ValidationError as e:
                return VerifierResult(
                    passed=False,
                    failure_type=FailureType.SCHEMA_VIOLATION,
                    failure_message=str(e.message),
                    confidence=1.0,
                    role=Role.DROP,
                    details={"schema_error": str(e)},
                )

        # 2. 必需字段检查
        missing_fields = self.REQUIRED_FIELDS - set(ums_json.keys())
        if missing_fields:
            return VerifierResult(
                passed=False,
                failure_type=FailureType.MISSING_FIELD,
                failure_message=f"Missing required fields: {missing_fields}",
                confidence=1.0,
                role=Role.DROP,
                details={"missing_fields": list(missing_fields)},
            )

        # 3. Modality 校验
        if ums_json.get("modality") not in self.VALID_MODALITIES:
            return VerifierResult(
                passed=False,
                failure_type=FailureType.ENUM_ERROR,
                failure_message=f"Invalid modality: {ums_json.get('modality')}",
                confidence=1.0,
                role=Role.DROP,
                details={"field": "modality", "value": ums_json.get("modality")},
            )

        # 4. Laterality 校验
        laterality = ums_json.get("laterality")
        if laterality not in self.VALID_LATERALITIES:
            return VerifierResult(
                passed=False,
                failure_type=FailureType.LATERALITY_ERROR,
                failure_message=f"Invalid laterality: {laterality}",
                confidence=1.0,
                role=Role.NEGATIVE_ONLY,
                details={"field": "laterality", "value": laterality},
            )

        # 5. Study view 校验
        study_view = ums_json.get("study_view")
        if study_view not in self.VALID_STUDY_VIEWS:
            return VerifierResult(
                passed=False,
                failure_type=FailureType.VIEW_ERROR,
                failure_message=f"Invalid study_view: {study_view}",
                confidence=1.0,
                role=Role.NEGATIVE_ONLY,
                details={"field": "study_view", "value": study_view},
            )

        # 6. Findings 校验
        findings = ums_json.get("findings", {})
        for name, finding in findings.items():
            if isinstance(finding, dict):
                state = finding.get("state")
                if state not in self.VALID_FINDING_STATES:
                    return VerifierResult(
                        passed=False,
                        failure_type=FailureType.ENUM_ERROR,
                        failure_message=f"Invalid finding state for {name}: {state}",
                        confidence=1.0,
                        role=Role.NEGATIVE_ONLY,
                        details={"field": f"findings.{name}.state", "value": state},
                    )

        # 7. Answerability 一致性检查
        answerability = ums_json.get("answerability", {})
        for field, answerable in answerability.items():
            if not answerable:  # 如果不可答
                # 检查对应字段是否为 null
                if field in findings:
                    finding_state = findings[field].get("state") if isinstance(findings[field], dict) else None
                    if finding_state is not None:
                        return VerifierResult(
                            passed=False,
                            failure_type=FailureType.ANSWERABILITY_INCONSISTENCY,
                            failure_message=f"Field {field} marked as unanswerable but has non-null value",
                            confidence=0.9,
                            role=Role.ABSTAIN,
                            details={"field": field, "answerable": answerable, "value": finding_state},
                        )

        # 8. Measurements 校验
        measurements = ums_json.get("measurements", {})
        for name, measurement in measurements.items():
            if isinstance(measurement, dict):
                # Unit 校验
                unit = measurement.get("unit")
                if unit not in self.VALID_UNITS:
                    return VerifierResult(
                        passed=False,
                        failure_type=FailureType.UNIT_ERROR,
                        failure_message=f"Invalid unit for {name}: {unit}",
                        confidence=1.0,
                        role=Role.NEGATIVE_ONLY,
                        details={"field": f"measurements.{name}.unit", "value": unit},
                    )

                # Range 校验
                value = measurement.get("value")
                if value is not None and self.strict_mode:
                    for range_name, (min_val, max_val) in self.MEASUREMENT_RANGES.items():
                        if range_name in name.lower():
                            if not (min_val <= value <= max_val):
                                return VerifierResult(
                                    passed=False,
                                    failure_type=FailureType.RANGE_ERROR,
                                    failure_message=f"Value out of range for {name}: {value}",
                                    confidence=0.8,
                                    role=Role.NEGATIVE_ONLY,
                                    details={
                                        "field": f"measurements.{name}.value",
                                        "value": value,
                                        "expected_range": (min_val, max_val),
                                    },
                                )

        # 9. Geometry 校验
        geometry = ums_json.get("geometry", {})
        bbox = geometry.get("bbox")
        if bbox is not None:
            if not (isinstance(bbox, list) and len(bbox) == 4):
                return VerifierResult(
                    passed=False,
                    failure_type=FailureType.GEOMETRY_MISMATCH,
                    failure_message=f"Invalid bbox format: {bbox}",
                    confidence=1.0,
                    role=Role.DROP,
                    details={"field": "geometry.bbox", "value": bbox},
                )
            # 检查坐标合法性
            if any(v < 0 for v in bbox):
                return VerifierResult(
                    passed=False,
                    failure_type=FailureType.GEOMETRY_MISMATCH,
                    failure_message=f"Negative bbox coordinates: {bbox}",
                    confidence=1.0,
                    role=Role.NEGATIVE_ONLY,
                    details={"field": "geometry.bbox", "value": bbox},
                )

        # 10. 确定 role
        # 检查是否有不可答字段
        has_unanswerable = any(not v for v in answerability.values())

        if has_unanswerable:
            role = Role.ABSTAIN
        else:
            role = Role.POSITIVE

        # 全部通过
        return VerifierResult(
            passed=True,
            failure_type=FailureType.NONE,
            failure_message="",
            confidence=1.0,
            role=role,
            details=details,
        )

    def verify_json_string(self, json_string: str) -> VerifierResult:
        """
        验证 JSON 字符串

        Args:
            json_string: JSON 字符串

        Returns:
            VerifierResult
        """
        try:
            ums_json = json.loads(json_string)
        except json.JSONDecodeError as e:
            return VerifierResult(
                passed=False,
                failure_type=FailureType.INVALID_JSON,
                failure_message=f"Invalid JSON: {str(e)}",
                confidence=1.0,
                role=Role.DROP,
                details={"json_error": str(e)},
            )

        return self.verify(ums_json)

    def batch_verify(self, ums_jsons: List[Dict[str, Any]]) -> List[VerifierResult]:
        """批量验证"""
        return [self.verify(ums_json) for ums_json in ums_jsons]

    def get_statistics(self, results: List[VerifierResult]) -> Dict[str, Any]:
        """
        获取验证统计

        Returns:
            包含通过率、失败类型分布等的字典
        """
        total = len(results)
        passed = sum(1 for r in results if r.passed)

        # 失败类型统计
        failure_counts = {}
        for r in results:
            if not r.passed:
                ft = r.failure_type.value
                failure_counts[ft] = failure_counts.get(ft, 0) + 1

        # Role 统计
        role_counts = {}
        for r in results:
            role = r.role.value
            role_counts[role] = role_counts.get(role, 0) + 1

        return {
            "total": total,
            "passed": passed,
            "pass_rate": passed / total if total > 0 else 0,
            "failure_counts": failure_counts,
            "role_counts": role_counts,
        }

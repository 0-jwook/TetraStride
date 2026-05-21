"""
커리큘럼 단계별 설정 정의.
각 Stage는 이전 Stage의 체크포인트에서 전이학습.

Stage 1: Standing v3c  — 안정 서있기 (height exp 보상)
Stage 2: Trot in-place — 제자리 trot
Stage 3: Forward walk  — 전진 보행
Stage 4: All-direction — 전방향 + 회전
"""

STAGES = {
    "standing_v3c": {
        "experiment_name": "spot_micro_stance_v3c",
        "max_iterations": 2000,
        "success_criteria": {
            "diag/term_ratio": ("lt", 0.005),
            "rew/upright": ("gt", 4.5),
            "diag/body_height_mean": ("range", 0.155, 0.185),
            "diag/left_span": ("lt", 0.015),
            "diag/right_span": ("lt", 0.015),
            "pose/shoulder_abs_mean": ("lt", 0.05),
        },
    },
    "trot_inplace": {
        "experiment_name": "spot_micro_trot_inplace",
        "max_iterations": 2000,
        "success_criteria": {
            "diag/term_ratio": ("lt", 0.01),
            "rew/gait": ("gt", 10.0),
            "diag/foot_tip_z_mean": ("gt", 0.025),
            "pose/stance_4_ratio": ("gt", 0.35),
        },
    },
    "forward_walk": {
        "experiment_name": "spot_micro_walk_fwd",
        "max_iterations": 3000,
        "success_criteria": {
            "diag/term_ratio": ("lt", 0.03),
            "diag/actual_lin_vel_x": ("gt", 0.20),
            "cmd/vel_tracking_err": ("lt", 0.12),
        },
    },
    "alldir_walk": {
        "experiment_name": "spot_micro_walk_alldir",
        "max_iterations": 3000,
        "success_criteria": {
            "diag/term_ratio": ("lt", 0.05),
            "cmd/vel_tracking_err": ("lt", 0.15),
        },
    },
}

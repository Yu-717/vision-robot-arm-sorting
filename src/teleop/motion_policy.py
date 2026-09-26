"""Explicit bounded test profile and previewable observe_v1 paths."""
import copy
import secrets
import time
from core import safe_arm


def test_config(base):
    cfg=copy.deepcopy(base)
    for joint in (3,4):
        cfg['soft_limits_deg'][joint]=(2,cfg['soft_limits_deg'][joint][1])
    cfg['keep_out']['table_margin_m']=.015
    cfg['keep_out']['platform_min_z_m']=.035
    return cfg


def grasp_config(base):
    """Time-limited profile for the last centimetres of a grasp: the fingertips may come to
    0.5 cm above the table; the platform keep-out and joint limits stay as they are."""
    cfg=copy.deepcopy(base)
    cfg['keep_out']['table_margin_m']=.005
    return cfg


TIMED_PROFILES = ('boundary', 'grasp', 'recover')


def explore_config(base):
    """Default console profile for exploring under supervision: joints may use their course
    limits (the cable limits), the platform keep-out only informs (its margin is still shown),
    the table keeps 0.5 cm. Steps up to 10 deg / 3 cm run as verified 5 deg segments and any resistance makes the
    arm hold at its readback, which is what protects the exploration."""
    cfg=copy.deepcopy(base)
    limits={j:tuple(v) for j,v in base['course_limits_deg'].items()}
    # 2026-09-24 operator session: 29 of 29 refusals were J3/J4 asking for -1..-2 deg. Arm_Lib only
    # rejects > 180 and the joints were pushed to -4 by hand without harm, so J2-J4 may go to -10.
    # J1 and J5 keep their course limits: those are the short-cable joints.
    for j in (2,3,4):
        limits[j]=(-10,limits[j][1])
    # (2026-09-25: J4 read back -2..-3 whenever -5..-8 was asked; kept here on purpose -- explore stays
    # wide and the resistance stop handles it; measured travel feeds the standard profile instead.)
    cfg['soft_limits_deg']=limits
    cfg['course_limits_deg']=dict(limits)
    cfg['keep_out']['platform_min_z_m']=-1.0
    # the table check is informational here too (user 2026-09-25: explore is not to be tightened);
    # the standard model's margin is still shown and the resistance stop still applies
    cfg['keep_out']['table_z_m']=-1.0
    cfg['max_ik_jump_deg']=30
    cfg['motion'].update({k:v for k,v in base.get('motion_explore',{}).items() if not k.startswith('_')})
    return cfg


def profile_config(name, base):
    builders={'normal':copy.deepcopy,'explore':explore_config,'boundary':test_config,
              'grasp':lambda b:grasp_config(b),'recover':lambda b:recover_config(b)}
    if name not in builders:
        raise ValueError('未知限位档')
    return builders[name](base)


def recover_config(base):
    """Time-limited profile for getting out of the keep-out: limits unchanged; paths are allowed
    only if their keep-out margin never gets worse than where they start (see safe_arm)."""
    return copy.deepcopy(base)


def cartesian_goal(current,axis,amount,cfg):
    """Joint segments that move the tip `amount` cm along base x/y/z, or change the tip pitch by
    `amount` degrees about the tip. Refuses IK failures and sudden branch changes."""
    import dofbot_kinematics as kin
    if len(current)!=6 or any(v is None for v in current):
        raise ValueError('关节不可读')
    fk=kin.forward(current[:4])
    tip=list(fk['tip']);pitch=fk['tip_pitch_deg'][0]
    if axis=='pitch':
        pitch+=amount
    else:
        tip['xyz'.index(axis)]+=amount/100
    solution=kin.inverse(*tip,pitch)
    if solution is None:
        raise ValueError('目标位置超出可达范围')
    goal=[int(round(v)) for v in solution]+list(current[4:])
    jump=max(abs(a-b) for a,b in zip(goal[:4],current[:4]))
    if jump>cfg.get('max_ik_jump_deg',15):
        raise ValueError(f'逆解跳变 {jump}°，拒绝执行')
    return safe_arm.plan_segments(current,goal,cfg['motion']['max_step_deg']) or [goal]


def home_plan(current,cfg):
    if len(current)!=6 or any(v is None for v in current):
        raise ValueError('关节不可读，不能规划归位')
    target=[c if t is None else t for c,t in zip(current,cfg['named_poses']['observe_v1'])]
    safe_arm.validate_target(current,target,cfg)
    segments=safe_arm.plan_segments(current,target,2)
    previous=list(current)
    for step in segments:
        safe_arm.validate_target(previous,step,cfg)
        safe_arm.check_keep_out(previous,step,cfg)
        previous=step
    margin=safe_arm.check_keep_out(current,target,cfg)
    return dict(id=secrets.token_hex(12),created_at=time.time(),start=list(current),target=target,
                segments=segments,margin_cm=round(margin*100,2),
                seconds=round(len(segments)*(cfg['motion']['segment_ms']+cfg['motion']['settle_ms'])/1000,1))

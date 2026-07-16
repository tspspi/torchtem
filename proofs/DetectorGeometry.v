Require Import Reals.
Require Import Psatz.

Open Scope R_scope.

Definition on_axis_radius (ay ax : R) : R :=
  sqrt (ay ^ 2 + ax ^ 2).

Definition shifted_radius (ay ax dy dx : R) : R :=
  sqrt ((ay - dy) ^ 2 + (ax - dx) ^ 2).

Definition annular_accepts (r inner outer : R) : Prop :=
  inner <= r < outer.

Lemma shifted_radius_zero_offset :
  forall ay ax : R,
    shifted_radius ay ax 0 0 = on_axis_radius ay ax.
Proof.
  intros ay ax.
  unfold shifted_radius, on_axis_radius.
  replace (ay - 0) with ay by lra.
  replace (ax - 0) with ax by lra.
  reflexivity.
Qed.

Lemma annular_accepts_zero_offset_equiv :
  forall ay ax inner outer : R,
    annular_accepts (shifted_radius ay ax 0 0) inner outer <->
    annular_accepts (on_axis_radius ay ax) inner outer.
Proof.
  intros ay ax inner outer.
  rewrite shifted_radius_zero_offset.
  tauto.
Qed.

Lemma shifted_radius_nonnegative :
  forall ay ax dy dx : R,
    0 <= shifted_radius ay ax dy dx.
Proof.
  intros ay ax dy dx.
  unfold shifted_radius.
  apply sqrt_pos.
Qed.


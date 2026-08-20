Require Import Reals.
Require Import Psatz.

Open Scope R_scope.

Definition affine_interp (start_A end_A t : R) : R :=
  start_A + (end_A - start_A) * t.

Lemma affine_interp_zero :
  forall start_A end_A : R,
    affine_interp start_A end_A 0 = start_A.
Proof.
  intros.
  unfold affine_interp.
  lra.
Qed.

Lemma affine_interp_one :
  forall start_A end_A : R,
    affine_interp start_A end_A 1 = end_A.
Proof.
  intros.
  unfold affine_interp.
  lra.
Qed.

Lemma affine_interp_constant :
  forall start_A t : R,
    affine_interp start_A start_A t = start_A.
Proof.
  intros.
  unfold affine_interp.
  lra.
Qed.

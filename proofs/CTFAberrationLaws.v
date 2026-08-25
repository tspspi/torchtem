Require Import Reals.
Require Import Psatz.

Open Scope R_scope.

Definition aberration_phase
  (wavelength alpha phi
   C10 C12 phi12
   C21 phi21 C23 phi23
   C30 C32 phi32 C34 phi34
   C41 phi41 C43 phi43 C45 phi45
   C50 C52 phi52 C54 phi54 C56 phi56 : R) : R :=
  (2 * PI / wavelength) *
    (0.5 * alpha ^ 2 * (C10 + C12 * cos (2 * (phi - phi12))) +
     (alpha ^ 3 / 3) *
       (C21 * cos (phi - phi21) +
        C23 * cos (3 * (phi - phi23))) +
     (alpha ^ 4 / 4) *
       (C30 +
        C32 * cos (2 * (phi - phi32)) +
        C34 * cos (4 * (phi - phi34))) +
     (alpha ^ 5 / 5) *
       (C41 * cos (phi - phi41) +
        C43 * cos (3 * (phi - phi43)) +
        C45 * cos (5 * (phi - phi45))) +
     (alpha ^ 6 / 6) *
       (C50 +
        C52 * cos (2 * (phi - phi52)) +
        C54 * cos (4 * (phi - phi54)) +
        C56 * cos (6 * (phi - phi56)))).

Lemma aberration_phase_zero_angle :
  forall wavelength phi
         C10 C12 phi12
         C21 phi21 C23 phi23
         C30 C32 phi32 C34 phi34
         C41 phi41 C43 phi43 C45 phi45
         C50 C52 phi52 C54 phi54 C56 phi56,
    aberration_phase wavelength 0 phi
      C10 C12 phi12
      C21 phi21 C23 phi23
      C30 C32 phi32 C34 phi34
      C41 phi41 C43 phi43 C45 phi45
      C50 C52 phi52 C54 phi54 C56 phi56 = 0.
Proof.
  intros.
  unfold aberration_phase.
  cbn.
  nra.
Qed.

Lemma aberration_phase_zero_coefficients :
  forall wavelength alpha phi
         phi12 phi21 phi23 phi32 phi34 phi41 phi43 phi45 phi52 phi54 phi56,
    aberration_phase wavelength alpha phi
      0 0 phi12
      0 phi21 0 phi23
      0 0 phi32 0 phi34
      0 phi41 0 phi43 0 phi45
      0 0 phi52 0 phi54 0 phi56 = 0.
Proof.
  intros.
  unfold aberration_phase.
  nra.
Qed.

Lemma aberration_phase_defocus_only :
  forall wavelength alpha phi
         phi12 phi21 phi23 phi32 phi34 phi41 phi43 phi45 phi52 phi54 phi56
         C10,
    aberration_phase wavelength alpha phi
      C10 0 phi12
      0 phi21 0 phi23
      0 0 phi32 0 phi34
      0 phi41 0 phi43 0 phi45
      0 0 phi52 0 phi54 0 phi56 =
    (2 * PI / wavelength) * (0.5 * alpha ^ 2 * C10).
Proof.
  intros.
  unfold aberration_phase.
  nra.
Qed.

Lemma aberration_phase_second_order_only :
  forall wavelength alpha phi
         C10 C12 phi12
         phi21 phi23 phi32 phi34 phi41 phi43 phi45 phi52 phi54 phi56,
    aberration_phase wavelength alpha phi
      C10 C12 phi12
      0 phi21 0 phi23
      0 0 phi32 0 phi34
      0 phi41 0 phi43 0 phi45
      0 0 phi52 0 phi54 0 phi56 =
    (2 * PI / wavelength) *
      (0.5 * alpha ^ 2 * (C10 + C12 * cos (2 * (phi - phi12)))).
Proof.
  intros.
  unfold aberration_phase.
  nra.
Qed.

Lemma aberration_phase_isotropic_only :
  forall wavelength alpha phi
         phi12 phi21 phi23 phi32 phi34 phi41 phi43 phi45 phi52 phi54 phi56
         C10 C30 C50,
    aberration_phase wavelength alpha phi
      C10 0 phi12
      0 phi21 0 phi23
      C30 0 phi32 0 phi34
      0 phi41 0 phi43 0 phi45
      C50 0 phi52 0 phi54 0 phi56 =
    (2 * PI / wavelength) *
      (0.5 * alpha ^ 2 * C10 +
       (alpha ^ 4 / 4) * C30 +
       (alpha ^ 6 / 6) * C50).
Proof.
  intros.
  unfold aberration_phase.
  nra.
Qed.

Lemma aberration_phase_third_order_only :
  forall wavelength alpha phi
         phi12 phi32 phi34 phi41 phi43 phi45 phi52 phi54 phi56
         C21 phi21 C23 phi23,
    aberration_phase wavelength alpha phi
      0 0 phi12
      C21 phi21 C23 phi23
      0 0 phi32 0 phi34
      0 phi41 0 phi43 0 phi45
      0 0 phi52 0 phi54 0 phi56 =
    (2 * PI / wavelength) *
      ((alpha ^ 3 / 3) *
         (C21 * cos (phi - phi21) +
          C23 * cos (3 * (phi - phi23)))).
Proof.
  intros.
  unfold aberration_phase.
  nra.
Qed.

Lemma aberration_phase_fourth_order_only :
  forall wavelength alpha phi
         phi12 phi21 phi23 phi41 phi43 phi45 phi52 phi54 phi56
         C30 C32 phi32 C34 phi34,
    aberration_phase wavelength alpha phi
      0 0 phi12
      0 phi21 0 phi23
      C30 C32 phi32 C34 phi34
      0 phi41 0 phi43 0 phi45
      0 0 phi52 0 phi54 0 phi56 =
    (2 * PI / wavelength) *
      ((alpha ^ 4 / 4) *
         (C30 +
          C32 * cos (2 * (phi - phi32)) +
          C34 * cos (4 * (phi - phi34)))).
Proof.
  intros.
  unfold aberration_phase.
  nra.
Qed.

Lemma aberration_phase_fifth_order_only :
  forall wavelength alpha phi
         phi12 phi21 phi23 phi32 phi34 phi52 phi54 phi56
         C41 phi41 C43 phi43 C45 phi45,
    aberration_phase wavelength alpha phi
      0 0 phi12
      0 phi21 0 phi23
      0 0 phi32 0 phi34
      C41 phi41 C43 phi43 C45 phi45
      0 0 phi52 0 phi54 0 phi56 =
    (2 * PI / wavelength) *
      ((alpha ^ 5 / 5) *
         (C41 * cos (phi - phi41) +
          C43 * cos (3 * (phi - phi43)) +
          C45 * cos (5 * (phi - phi45)))).
Proof.
  intros.
  unfold aberration_phase.
  nra.
Qed.

Lemma aberration_phase_sixth_order_only :
  forall wavelength alpha phi
         phi12 phi21 phi23 phi32 phi34 phi41 phi43 phi45
         C50 C52 phi52 C54 phi54 C56 phi56,
    aberration_phase wavelength alpha phi
      0 0 phi12
      0 phi21 0 phi23
      0 0 phi32 0 phi34
      0 phi41 0 phi43 0 phi45
      C50 C52 phi52 C54 phi54 C56 phi56 =
    (2 * PI / wavelength) *
      ((alpha ^ 6 / 6) *
         (C50 +
          C52 * cos (2 * (phi - phi52)) +
          C54 * cos (4 * (phi - phi54)) +
          C56 * cos (6 * (phi - phi56)))).
Proof.
  intros.
  unfold aberration_phase.
  nra.
Qed.

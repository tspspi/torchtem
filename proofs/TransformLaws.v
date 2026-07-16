Require Import Reals.
Require Import FunctionalExtensionality.

Open Scope R_scope.

Definition transform := R -> R.

Definition identity_transform : transform := fun x => x.

Definition compose (g f : transform) : transform :=
  fun x => g (f x).

Lemma compose_identity_left :
  forall f : transform,
    compose identity_transform f = f.
Proof.
  intro f.
  unfold compose, identity_transform.
  apply functional_extensionality.
  intro x.
  reflexivity.
Qed.

Lemma compose_identity_right :
  forall f : transform,
    compose f identity_transform = f.
Proof.
  intro f.
  unfold compose, identity_transform.
  apply functional_extensionality.
  intro x.
  reflexivity.
Qed.

Lemma compose_associative :
  forall f g h : transform,
    compose h (compose g f) = compose (compose h g) f.
Proof.
  intros f g h.
  unfold compose.
  apply functional_extensionality.
  intro x.
  reflexivity.
Qed.


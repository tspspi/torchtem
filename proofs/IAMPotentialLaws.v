Require Import List.
Require Import Reals.
Require Import Psatz.

Import ListNotations.
Open Scope R_scope.

Definition z_in_slice (z a b : R) : Prop :=
  a <= z < b.

Lemma z_in_slice_left_inclusive :
  forall a b : R,
    a < b ->
    z_in_slice a a b.
Proof.
  intros a b Hab.
  unfold z_in_slice.
  lra.
Qed.

Lemma z_in_slice_right_exclusive :
  forall z a b : R,
    z_in_slice z a b ->
    z <> b.
Proof.
  intros z a b Hz.
  unfold z_in_slice in Hz.
  lra.
Qed.

Lemma adjacent_slices_disjoint :
  forall z a b c : R,
    b <= c ->
    z_in_slice z a b ->
    z_in_slice z c c ->
    False.
Proof.
  intros z a b c Hbc Hzab Hzcc.
  unfold z_in_slice in *.
  lra.
Qed.

Lemma separated_slices_disjoint :
  forall z a b c d : R,
    b <= c ->
    z_in_slice z a b ->
    z_in_slice z c d ->
    False.
Proof.
  intros z a b c d Hbc Hzab Hzcd.
  unfold z_in_slice in *.
  lra.
Qed.

Section RenderLaws.

Variable Slice : Type.
Variable render_infinite_slice : R -> R -> Slice.
Variable render_finite_slice : R -> R -> Slice.

Inductive projection_mode :=
| InfiniteProjection
| FiniteProjection.

Fixpoint render_slices
  (mode : projection_mode)
  (limits : list (R * R)) : list Slice :=
  match limits with
  | [] => []
  | (a, b) :: rest =>
      match mode with
      | InfiniteProjection => render_infinite_slice a b
      | FiniteProjection => render_finite_slice a b
      end :: render_slices mode rest
  end.

Lemma render_slices_length :
  forall mode limits,
    length (render_slices mode limits) = length limits.
Proof.
  intros mode limits.
  induction limits as [|[a b] rest IH].
  - reflexivity.
  - simpl. rewrite IH. reflexivity.
Qed.

Lemma render_slices_infinite_head :
  forall a b rest,
    render_slices InfiniteProjection ((a, b) :: rest) =
    render_infinite_slice a b :: render_slices InfiniteProjection rest.
Proof.
  reflexivity.
Qed.

Lemma render_slices_finite_head :
  forall a b rest,
    render_slices FiniteProjection ((a, b) :: rest) =
    render_finite_slice a b :: render_slices FiniteProjection rest.
Proof.
  reflexivity.
Qed.

End RenderLaws.

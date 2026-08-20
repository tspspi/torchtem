Require Import List.
Require Import Reals.
Require Import Psatz.
Require Import Lia.

Import ListNotations.
Open Scope R_scope.

Fixpoint sum_list (xs : list R) : R :=
  match xs with
  | [] => 0
  | x :: rest => x + sum_list rest
  end.

Definition expand_equal_slices (num_slices : nat) (potential : R) : list R :=
  match num_slices with
  | O => []
  | S _ => repeat (potential / INR num_slices) num_slices
  end.

Definition potential_slices_model
  (already_sliced : bool)
  (num_slices : nat)
  (slices : list R)
  (potential : R) : list R :=
  if already_sliced then slices else expand_equal_slices num_slices potential.

Lemma expand_equal_slices_length :
  forall num_slices potential,
    length (expand_equal_slices num_slices potential) = num_slices.
Proof.
  intros num_slices potential.
  destruct num_slices as [|n].
  - reflexivity.
  - unfold expand_equal_slices. simpl. rewrite repeat_length. reflexivity.
Qed.

Lemma sum_list_repeat :
  forall x n,
    sum_list (repeat x n) = x * INR n.
Proof.
  intros x n.
  induction n as [|n IH].
  - simpl. lra.
  - simpl. rewrite IH.
    destruct n.
    + simpl. nra.
    + simpl. nra.
Qed.

Lemma expand_equal_slices_preserves_total :
  forall num_slices potential,
    (0 < num_slices)%nat ->
    sum_list (expand_equal_slices num_slices potential) = potential.
Proof.
  intros num_slices potential Hpos.
  destruct num_slices as [|n].
  - lia.
  - unfold expand_equal_slices.
    rewrite sum_list_repeat.
    assert (Hnz : INR (S n) <> 0).
    { apply not_0_INR. lia. }
    field.
    exact Hnz.
Qed.

Lemma potential_slices_model_preserves_existing_slices :
  forall num_slices slices potential,
    potential_slices_model true num_slices slices potential = slices.
Proof.
  reflexivity.
Qed.

Lemma potential_slices_model_expands_unsliced_plane :
  forall num_slices slices potential,
    potential_slices_model false num_slices slices potential =
    expand_equal_slices num_slices potential.
Proof.
  reflexivity.
Qed.

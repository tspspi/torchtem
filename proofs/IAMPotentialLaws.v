Require Import List.
Require Import Reals.
Require Import Psatz.
Require Import Lia.

Import ListNotations.
Open Scope R_scope.

Definition z_in_slice (z a b : R) : Prop :=
  a <= z < b.

Fixpoint sum_list (xs : list R) : R :=
  match xs with
  | [] => 0
  | x :: rest => x + sum_list rest
  end.

Fixpoint slice_limits_from (start : R) (slice_thickness : list R) : list (R * R) :=
  match slice_thickness with
  | [] => []
  | d :: rest =>
      let stop := start + d in
      (start, stop) :: slice_limits_from stop rest
  end.

Fixpoint final_edge_from (start : R) (slice_thickness : list R) : R :=
  match slice_thickness with
  | [] => start
  | d :: rest => final_edge_from (start + d) rest
  end.

Definition slice_limits_model (slice_thickness : list R) : list (R * R) :=
  slice_limits_from 0 slice_thickness.

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

Lemma slice_limits_model_nil :
  slice_limits_model [] = [].
Proof.
  reflexivity.
Qed.

Lemma slice_limits_from_length :
  forall start thicknesses,
    length (slice_limits_from start thicknesses) = length thicknesses.
Proof.
  intros start thicknesses.
  revert start.
  induction thicknesses as [|d rest IH]; intros start.
  - reflexivity.
  - simpl. rewrite IH. reflexivity.
Qed.

Lemma slice_limits_model_length :
  forall thicknesses,
    length (slice_limits_model thicknesses) = length thicknesses.
Proof.
  intros thicknesses.
  unfold slice_limits_model.
  apply slice_limits_from_length.
Qed.

Lemma slice_limits_model_head :
  forall d rest,
    slice_limits_model (d :: rest) =
    (0, d) :: slice_limits_from d rest.
Proof.
  intros d rest.
  unfold slice_limits_model.
  simpl.
  replace (0 + d) with d by lra.
  reflexivity.
Qed.

Lemma slice_limits_model_singleton :
  forall d,
    slice_limits_model [d] = [(0, d)].
Proof.
  intro d.
  unfold slice_limits_model.
  simpl.
  replace (0 + d) with d by lra.
  reflexivity.
Qed.

Lemma slice_limits_from_adjacent :
  forall start d1 d2 rest,
    slice_limits_from start (d1 :: d2 :: rest) =
    (start, start + d1) :: (start + d1, start + d1 + d2) :: slice_limits_from (start + d1 + d2) rest.
Proof.
  reflexivity.
Qed.

Lemma slice_limits_model_adjacent_touch :
  forall d1 d2 rest,
    slice_limits_model (d1 :: d2 :: rest) =
    (0, d1) :: (d1, d1 + d2) :: slice_limits_from (d1 + d2) rest.
Proof.
  intros d1 d2 rest.
  unfold slice_limits_model.
  simpl.
  replace (0 + d1) with d1 by lra.
  replace (d1 + d2) with (d1 + d2) by lra.
  reflexivity.
Qed.

Lemma slice_limits_model_two_slices :
  forall d1 d2,
    slice_limits_model [d1; d2] = [(0, d1); (d1, d1 + d2)].
Proof.
  intros d1 d2.
  unfold slice_limits_model.
  simpl.
  replace (0 + d1) with d1 by lra.
  replace (d1 + d2) with (d1 + d2) by lra.
  reflexivity.
Qed.

Lemma slice_limits_model_adjacent_membership_disjoint :
  forall z d1 d2,
    z_in_slice z 0 d1 ->
    z_in_slice z d1 (d1 + d2) ->
    False.
Proof.
  intros z d1 d2 Hz1 Hz2.
  apply (separated_slices_disjoint z 0 d1 d1 (d1 + d2)).
  - lra.
  - exact Hz1.
  - exact Hz2.
Qed.

Lemma final_edge_from_preserves_total :
  forall start thicknesses,
    final_edge_from start thicknesses = start + sum_list thicknesses.
Proof.
  intros start thicknesses.
  revert start.
  induction thicknesses as [|d rest IH]; intros start.
  - simpl. lra.
  - simpl.
    rewrite IH.
    lra.
Qed.

Lemma final_edge_from_zero :
  forall thicknesses,
    final_edge_from 0 thicknesses = sum_list thicknesses.
Proof.
  intro thicknesses.
  rewrite final_edge_from_preserves_total.
  lra.
Qed.

Lemma final_edge_from_app :
  forall start xs ys,
    final_edge_from start (xs ++ ys) =
    final_edge_from (final_edge_from start xs) ys.
Proof.
  intros start xs.
  revert start.
  induction xs as [|d rest IH]; intros start ys.
  - reflexivity.
  - simpl.
    apply IH.
Qed.

Lemma sum_list_app :
  forall xs ys,
    sum_list (xs ++ ys) = sum_list xs + sum_list ys.
Proof.
  intros xs ys.
  induction xs as [|x rest IH].
  - simpl. lra.
  - simpl. rewrite IH. lra.
Qed.

Lemma sum_list_app_singleton :
  forall xs d,
    sum_list (xs ++ [d]) = sum_list xs + d.
Proof.
  intros xs d.
  rewrite sum_list_app.
  simpl.
  lra.
Qed.

Lemma slice_limits_from_app :
  forall start xs ys,
    slice_limits_from start (xs ++ ys) =
    slice_limits_from start xs ++
    slice_limits_from (final_edge_from start xs) ys.
Proof.
  intros start xs.
  revert start.
  induction xs as [|d rest IH]; intros start ys.
  - reflexivity.
  - simpl.
    rewrite IH.
    reflexivity.
Qed.

Lemma slice_limits_model_app :
  forall xs ys,
    slice_limits_model (xs ++ ys) =
    slice_limits_model xs ++
    slice_limits_from (final_edge_from 0 xs) ys.
Proof.
  intros xs ys.
  unfold slice_limits_model.
  apply slice_limits_from_app.
Qed.

Lemma slice_limits_model_length_app :
  forall xs ys,
    length (slice_limits_model (xs ++ ys)) =
    (length (slice_limits_model xs) + length (slice_limits_model ys))%nat.
Proof.
  intros xs ys.
  rewrite slice_limits_model_length.
  rewrite app_length.
  rewrite !slice_limits_model_length.
  reflexivity.
Qed.

Lemma slice_limits_from_app_singleton :
  forall start xs d,
    slice_limits_from start (xs ++ [d]) =
    slice_limits_from start xs ++
    [(final_edge_from start xs, final_edge_from start xs + d)].
Proof.
  intros start xs d.
  rewrite slice_limits_from_app.
  simpl.
  reflexivity.
Qed.

Lemma slice_limits_from_app_singleton_contains_last :
  forall start xs d,
    In (final_edge_from start xs, final_edge_from start xs + d)
       (slice_limits_from start (xs ++ [d])).
Proof.
  intros start xs d.
  rewrite slice_limits_from_app_singleton.
  apply in_or_app.
  right.
  simpl.
  left.
  reflexivity.
Qed.

Lemma slice_limits_from_app_singleton_uses_final_edge :
  forall start xs d,
    slice_limits_from start (xs ++ [d]) =
    slice_limits_from start xs ++
    [(final_edge_from start xs, final_edge_from start (xs ++ [d]))].
Proof.
  intros start xs d.
  rewrite slice_limits_from_app_singleton.
  rewrite final_edge_from_preserves_total.
  rewrite final_edge_from_preserves_total.
  rewrite sum_list_app_singleton.
  replace (start + (sum_list xs + d)) with (start + sum_list xs + d) by lra.
  reflexivity.
Qed.

Lemma slice_limits_from_app_singleton_contains_last_by_final_edge :
  forall start xs d,
    In (final_edge_from start xs, final_edge_from start (xs ++ [d]))
       (slice_limits_from start (xs ++ [d])).
Proof.
  intros start xs d.
  rewrite slice_limits_from_app_singleton_uses_final_edge.
  apply in_or_app.
  right.
  simpl.
  left.
  reflexivity.
Qed.

Lemma slice_limits_model_app_singleton :
  forall xs d,
    slice_limits_model (xs ++ [d]) =
    slice_limits_model xs ++
    [(sum_list xs, sum_list xs + d)].
Proof.
  intros xs d.
  rewrite slice_limits_model_app.
  rewrite final_edge_from_zero.
  simpl.
  reflexivity.
Qed.

Lemma final_edge_from_app_singleton :
  forall start xs d,
    final_edge_from start (xs ++ [d]) =
    final_edge_from start xs + d.
Proof.
  intros start xs d.
  rewrite final_edge_from_app.
  simpl.
  lra.
Qed.

Lemma slice_limits_model_app_singleton_contains_last :
  forall xs d,
    In (sum_list xs, sum_list xs + d) (slice_limits_model (xs ++ [d])).
Proof.
  intros xs d.
  rewrite slice_limits_model_app_singleton.
  apply in_or_app.
  right.
  simpl.
  left.
  reflexivity.
Qed.

Lemma slice_limits_model_app_singleton_last_ends_at_total :
  forall xs d,
    In (sum_list xs, sum_list (xs ++ [d])) (slice_limits_model (xs ++ [d])).
Proof.
  intros xs d.
  rewrite sum_list_app_singleton.
  apply slice_limits_model_app_singleton_contains_last.
Qed.

Lemma slice_limits_model_app_singleton_uses_final_edge :
  forall xs d,
    slice_limits_model (xs ++ [d]) =
    slice_limits_model xs ++
    [(final_edge_from 0 xs, final_edge_from 0 (xs ++ [d]))].
Proof.
  intros xs d.
  rewrite slice_limits_model_app_singleton.
  rewrite final_edge_from_zero.
  rewrite final_edge_from_app_singleton.
  rewrite final_edge_from_zero.
  reflexivity.
Qed.

Lemma slice_limits_model_app_singleton_contains_last_by_final_edge :
  forall xs d,
    In (final_edge_from 0 xs, final_edge_from 0 (xs ++ [d]))
       (slice_limits_model (xs ++ [d])).
Proof.
  intros xs d.
  rewrite slice_limits_model_app_singleton_uses_final_edge.
  apply in_or_app.
  right.
  simpl.
  left.
  reflexivity.
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

Lemma render_slices_app :
  forall mode xs ys,
    render_slices mode (xs ++ ys) =
    render_slices mode xs ++ render_slices mode ys.
Proof.
  intros mode xs.
  induction xs as [|[a b] rest IH]; intros ys.
  - reflexivity.
  - simpl. rewrite IH. reflexivity.
Qed.

Lemma render_slices_slice_limits_model_app :
  forall mode xs ys,
    render_slices mode (slice_limits_model (xs ++ ys)) =
    render_slices mode (slice_limits_model xs) ++
    render_slices mode (slice_limits_from (final_edge_from 0 xs) ys).
Proof.
  intros mode xs ys.
  rewrite slice_limits_model_app.
  apply render_slices_app.
Qed.

Lemma render_slices_slice_limits_model_length_app :
  forall mode xs ys,
    length (render_slices mode (slice_limits_model (xs ++ ys))) =
    (length (render_slices mode (slice_limits_model xs)) +
     length (render_slices mode (slice_limits_from (final_edge_from 0 xs) ys)))%nat.
Proof.
  intros mode xs ys.
  rewrite render_slices_slice_limits_model_app.
  rewrite app_length.
  reflexivity.
Qed.

Lemma render_slices_slice_limits_model_length :
  forall mode thicknesses,
    length (render_slices mode (slice_limits_model thicknesses)) = length thicknesses.
Proof.
  intros mode thicknesses.
  rewrite render_slices_length.
  apply slice_limits_model_length.
Qed.

Lemma render_slices_slice_limits_model_infinite_singleton :
  forall d,
    render_slices InfiniteProjection (slice_limits_model [d]) =
    [render_infinite_slice 0 d].
Proof.
  intro d.
  rewrite slice_limits_model_singleton.
  reflexivity.
Qed.

Lemma render_slices_slice_limits_model_finite_singleton :
  forall d,
    render_slices FiniteProjection (slice_limits_model [d]) =
    [render_finite_slice 0 d].
Proof.
  intro d.
  rewrite slice_limits_model_singleton.
  reflexivity.
Qed.

Lemma render_slices_slice_limits_model_infinite_two_slices :
  forall d1 d2,
    render_slices InfiniteProjection (slice_limits_model [d1; d2]) =
    [render_infinite_slice 0 d1; render_infinite_slice d1 (d1 + d2)].
Proof.
  intros d1 d2.
  rewrite slice_limits_model_two_slices.
  reflexivity.
Qed.

Lemma render_slices_slice_limits_model_finite_two_slices :
  forall d1 d2,
    render_slices FiniteProjection (slice_limits_model [d1; d2]) =
    [render_finite_slice 0 d1; render_finite_slice d1 (d1 + d2)].
Proof.
  intros d1 d2.
  rewrite slice_limits_model_two_slices.
  reflexivity.
Qed.

Lemma render_slices_slice_limits_model_infinite_app_singleton :
  forall xs d,
    render_slices InfiniteProjection (slice_limits_model (xs ++ [d])) =
    render_slices InfiniteProjection (slice_limits_model xs) ++
    [render_infinite_slice (sum_list xs) (sum_list xs + d)].
Proof.
  intros xs d.
  rewrite slice_limits_model_app_singleton.
  rewrite render_slices_app.
  reflexivity.
Qed.

Lemma render_slices_slice_limits_model_finite_app_singleton :
  forall xs d,
    render_slices FiniteProjection (slice_limits_model (xs ++ [d])) =
    render_slices FiniteProjection (slice_limits_model xs) ++
    [render_finite_slice (sum_list xs) (sum_list xs + d)].
Proof.
  intros xs d.
  rewrite slice_limits_model_app_singleton.
  rewrite render_slices_app.
  reflexivity.
Qed.

Lemma render_slices_slice_limits_model_infinite_app_singleton_ends_at_total :
  forall xs d,
    render_slices InfiniteProjection (slice_limits_model (xs ++ [d])) =
    render_slices InfiniteProjection (slice_limits_model xs) ++
    [render_infinite_slice (sum_list xs) (sum_list (xs ++ [d]))].
Proof.
  intros xs d.
  rewrite sum_list_app_singleton.
  apply render_slices_slice_limits_model_infinite_app_singleton.
Qed.

Lemma render_slices_slice_limits_model_finite_app_singleton_ends_at_total :
  forall xs d,
    render_slices FiniteProjection (slice_limits_model (xs ++ [d])) =
    render_slices FiniteProjection (slice_limits_model xs) ++
    [render_finite_slice (sum_list xs) (sum_list (xs ++ [d]))].
Proof.
  intros xs d.
  rewrite sum_list_app_singleton.
  apply render_slices_slice_limits_model_finite_app_singleton.
Qed.

Lemma render_slices_slice_limits_model_infinite_app_singleton_uses_final_edge :
  forall xs d,
    render_slices InfiniteProjection (slice_limits_model (xs ++ [d])) =
    render_slices InfiniteProjection (slice_limits_model xs) ++
    [render_infinite_slice (final_edge_from 0 xs) (final_edge_from 0 (xs ++ [d]))].
Proof.
  intros xs d.
  rewrite final_edge_from_zero.
  rewrite final_edge_from_zero.
  apply render_slices_slice_limits_model_infinite_app_singleton_ends_at_total.
Qed.

Lemma render_slices_slice_limits_model_finite_app_singleton_uses_final_edge :
  forall xs d,
    render_slices FiniteProjection (slice_limits_model (xs ++ [d])) =
    render_slices FiniteProjection (slice_limits_model xs) ++
    [render_finite_slice (final_edge_from 0 xs) (final_edge_from 0 (xs ++ [d]))].
Proof.
  intros xs d.
  rewrite final_edge_from_zero.
  rewrite final_edge_from_zero.
  apply render_slices_slice_limits_model_finite_app_singleton_ends_at_total.
Qed.

Lemma render_slices_slice_limits_model_infinite_app_singleton_contains_last :
  forall xs d,
    In (render_infinite_slice (sum_list xs) (sum_list (xs ++ [d])))
       (render_slices InfiniteProjection (slice_limits_model (xs ++ [d]))).
Proof.
  intros xs d.
  rewrite render_slices_slice_limits_model_infinite_app_singleton_ends_at_total.
  apply in_or_app.
  right.
  simpl.
  left.
  reflexivity.
Qed.

Lemma render_slices_slice_limits_model_finite_app_singleton_contains_last :
  forall xs d,
    In (render_finite_slice (sum_list xs) (sum_list (xs ++ [d])))
       (render_slices FiniteProjection (slice_limits_model (xs ++ [d]))).
Proof.
  intros xs d.
  rewrite render_slices_slice_limits_model_finite_app_singleton_ends_at_total.
  apply in_or_app.
  right.
  simpl.
  left.
  reflexivity.
Qed.

Lemma render_slices_slice_limits_model_infinite_app_singleton_contains_last_by_final_edge :
  forall xs d,
    In (render_infinite_slice (final_edge_from 0 xs) (final_edge_from 0 (xs ++ [d])))
       (render_slices InfiniteProjection (slice_limits_model (xs ++ [d]))).
Proof.
  intros xs d.
  rewrite render_slices_slice_limits_model_infinite_app_singleton_uses_final_edge.
  apply in_or_app.
  right.
  simpl.
  left.
  reflexivity.
Qed.

Lemma render_slices_slice_limits_model_finite_app_singleton_contains_last_by_final_edge :
  forall xs d,
    In (render_finite_slice (final_edge_from 0 xs) (final_edge_from 0 (xs ++ [d])))
       (render_slices FiniteProjection (slice_limits_model (xs ++ [d]))).
Proof.
  intros xs d.
  rewrite render_slices_slice_limits_model_finite_app_singleton_uses_final_edge.
  apply in_or_app.
  right.
  simpl.
  left.
  reflexivity.
Qed.

Lemma render_slices_infinite_slice_limits_from_app_singleton :
  forall start xs d,
    render_slices InfiniteProjection (slice_limits_from start (xs ++ [d])) =
    render_slices InfiniteProjection (slice_limits_from start xs) ++
    [render_infinite_slice (final_edge_from start xs) (final_edge_from start xs + d)].
Proof.
  intros start xs d.
  rewrite slice_limits_from_app_singleton.
  rewrite render_slices_app.
  reflexivity.
Qed.

Lemma render_slices_finite_slice_limits_from_app_singleton :
  forall start xs d,
    render_slices FiniteProjection (slice_limits_from start (xs ++ [d])) =
    render_slices FiniteProjection (slice_limits_from start xs) ++
    [render_finite_slice (final_edge_from start xs) (final_edge_from start xs + d)].
Proof.
  intros start xs d.
  rewrite slice_limits_from_app_singleton.
  rewrite render_slices_app.
  reflexivity.
Qed.

Lemma render_slices_infinite_slice_limits_from_app_singleton_uses_final_edge :
  forall start xs d,
    render_slices InfiniteProjection (slice_limits_from start (xs ++ [d])) =
    render_slices InfiniteProjection (slice_limits_from start xs) ++
    [render_infinite_slice (final_edge_from start xs) (final_edge_from start (xs ++ [d]))].
Proof.
  intros start xs d.
  rewrite render_slices_infinite_slice_limits_from_app_singleton.
  rewrite final_edge_from_app_singleton.
  reflexivity.
Qed.

Lemma render_slices_finite_slice_limits_from_app_singleton_uses_final_edge :
  forall start xs d,
    render_slices FiniteProjection (slice_limits_from start (xs ++ [d])) =
    render_slices FiniteProjection (slice_limits_from start xs) ++
    [render_finite_slice (final_edge_from start xs) (final_edge_from start (xs ++ [d]))].
Proof.
  intros start xs d.
  rewrite render_slices_finite_slice_limits_from_app_singleton.
  rewrite final_edge_from_app_singleton.
  reflexivity.
Qed.

Lemma render_slices_infinite_slice_limits_from_app_singleton_contains_last :
  forall start xs d,
    In (render_infinite_slice (final_edge_from start xs) (final_edge_from start xs + d))
       (render_slices InfiniteProjection (slice_limits_from start (xs ++ [d]))).
Proof.
  intros start xs d.
  rewrite render_slices_infinite_slice_limits_from_app_singleton.
  apply in_or_app.
  right.
  simpl.
  left.
  reflexivity.
Qed.

Lemma render_slices_finite_slice_limits_from_app_singleton_contains_last :
  forall start xs d,
    In (render_finite_slice (final_edge_from start xs) (final_edge_from start xs + d))
       (render_slices FiniteProjection (slice_limits_from start (xs ++ [d]))).
Proof.
  intros start xs d.
  rewrite render_slices_finite_slice_limits_from_app_singleton.
  apply in_or_app.
  right.
  simpl.
  left.
  reflexivity.
Qed.

Lemma render_slices_infinite_slice_limits_from_app_singleton_contains_last_by_final_edge :
  forall start xs d,
    In (render_infinite_slice (final_edge_from start xs) (final_edge_from start (xs ++ [d])))
       (render_slices InfiniteProjection (slice_limits_from start (xs ++ [d]))).
Proof.
  intros start xs d.
  rewrite render_slices_infinite_slice_limits_from_app_singleton_uses_final_edge.
  apply in_or_app.
  right.
  simpl.
  left.
  reflexivity.
Qed.

Lemma render_slices_finite_slice_limits_from_app_singleton_contains_last_by_final_edge :
  forall start xs d,
    In (render_finite_slice (final_edge_from start xs) (final_edge_from start (xs ++ [d])))
       (render_slices FiniteProjection (slice_limits_from start (xs ++ [d]))).
Proof.
  intros start xs d.
  rewrite render_slices_finite_slice_limits_from_app_singleton_uses_final_edge.
  apply in_or_app.
  right.
  simpl.
  left.
  reflexivity.
Qed.

Lemma render_slices_infinite_head :
  forall a b rest,
    render_slices InfiniteProjection ((a, b) :: rest) =
    render_infinite_slice a b :: render_slices InfiniteProjection rest.
Proof.
  reflexivity.
Qed.

Lemma render_slices_infinite_singleton :
  forall a b,
    render_slices InfiniteProjection [(a, b)] =
    [render_infinite_slice a b].
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

Lemma render_slices_finite_singleton :
  forall a b,
    render_slices FiniteProjection [(a, b)] =
    [render_finite_slice a b].
Proof.
  reflexivity.
Qed.

End RenderLaws.

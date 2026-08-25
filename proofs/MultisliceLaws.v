Require Import List.

Import ListNotations.

Section MultisliceLaws.

Variable Slice W : Type.
Variable transmit : Slice -> W -> W.
Variable propagate : W -> W.

Fixpoint multislice
  (incident : W)
  (slices : list Slice)
  (propagate_last : bool) : W :=
  match slices with
  | [] => incident
  | [slice] =>
      let wave := transmit slice incident in
      if propagate_last then propagate wave else wave
  | slice :: rest =>
      multislice (propagate (transmit slice incident)) rest propagate_last
  end.

Lemma multislice_nil :
  forall incident : W,
    multislice incident [] true = incident /\
    multislice incident [] false = incident.
Proof.
  intro incident.
  split; reflexivity.
Qed.

Lemma multislice_singleton_false :
  forall (incident : W) (slice : Slice),
    multislice incident [slice] false = transmit slice incident.
Proof.
  intros incident slice.
  reflexivity.
Qed.

Lemma multislice_singleton_true :
  forall (incident : W) (slice : Slice),
    multislice incident [slice] true = propagate (transmit slice incident).
Proof.
  intros incident slice.
  reflexivity.
Qed.

Lemma multislice_two_slices_false :
  forall (incident : W) (slice1 slice2 : Slice),
    multislice incident [slice1; slice2] false =
    transmit slice2 (propagate (transmit slice1 incident)).
Proof.
  intros incident slice1 slice2.
  reflexivity.
Qed.

Lemma multislice_two_slices_true :
  forall (incident : W) (slice1 slice2 : Slice),
    multislice incident [slice1; slice2] true =
    propagate (transmit slice2 (propagate (transmit slice1 incident))).
Proof.
  intros incident slice1 slice2.
  reflexivity.
Qed.

Lemma multislice_three_slices_false :
  forall (incident : W) (slice1 slice2 slice3 : Slice),
    multislice incident [slice1; slice2; slice3] false =
    transmit slice3
      (propagate
         (transmit slice2
            (propagate (transmit slice1 incident)))).
Proof.
  intros incident slice1 slice2 slice3.
  reflexivity.
Qed.

Lemma multislice_three_slices_true :
  forall (incident : W) (slice1 slice2 slice3 : Slice),
    multislice incident [slice1; slice2; slice3] true =
    propagate
      (transmit slice3
         (propagate
            (transmit slice2
               (propagate (transmit slice1 incident))))).
Proof.
  intros incident slice1 slice2 slice3.
  reflexivity.
Qed.

Lemma multislice_propagate_last_toggle :
  forall (incident : W) (slices : list Slice),
    slices <> [] ->
    multislice incident slices true =
    propagate (multislice incident slices false).
Proof.
  intros incident slices.
  revert incident.
  induction slices as [|slice rest IH]; intros incident Hslices.
  - exfalso. apply Hslices. reflexivity.
  - destruct rest as [|slice' rest'].
    + reflexivity.
    + simpl.
      apply IH.
      discriminate.
Qed.

Lemma multislice_app_true :
  forall (prefix suffix : list Slice) (incident : W),
    multislice incident (prefix ++ suffix) true =
    multislice (multislice incident prefix true) suffix true.
Proof.
  induction prefix as [|slice prefix IH]; intros suffix incident.
  - reflexivity.
  - destruct prefix as [|slice' prefix'].
    + simpl.
      destruct suffix as [|tail rest].
      * reflexivity.
      * reflexivity.
    + simpl.
      apply IH.
Qed.

Lemma multislice_app_false :
  forall (prefix suffix : list Slice) (incident : W),
    suffix <> [] ->
    multislice incident (prefix ++ suffix) false =
    multislice (multislice incident prefix true) suffix false.
Proof.
  induction prefix as [|slice prefix IH]; intros suffix incident Hsuffix.
  - reflexivity.
  - destruct prefix as [|slice' prefix'].
    + destruct suffix as [|tail rest].
      * exfalso. apply Hsuffix. reflexivity.
      * simpl. reflexivity.
    + simpl.
      apply IH.
      exact Hsuffix.
Qed.

Lemma multislice_app_true_nonempty_suffix :
  forall (prefix suffix : list Slice) (incident : W),
    suffix <> [] ->
    multislice incident (prefix ++ suffix) true =
    propagate (multislice (multislice incident prefix true) suffix false).
Proof.
  intros prefix suffix incident Hsuffix.
  rewrite multislice_app_true.
  apply multislice_propagate_last_toggle.
  exact Hsuffix.
Qed.

Lemma multislice_app_singleton_false :
  forall (prefix : list Slice) (incident : W) (slice : Slice),
    multislice incident (prefix ++ [slice]) false =
    transmit slice (multislice incident prefix true).
Proof.
  intros prefix incident slice.
  rewrite multislice_app_false.
  apply multislice_singleton_false.
  discriminate.
Qed.

Lemma multislice_app_singleton_true :
  forall (prefix : list Slice) (incident : W) (slice : Slice),
    multislice incident (prefix ++ [slice]) true =
    propagate (transmit slice (multislice incident prefix true)).
Proof.
  intros prefix incident slice.
  rewrite multislice_app_true.
  apply multislice_singleton_true.
Qed.

End MultisliceLaws.

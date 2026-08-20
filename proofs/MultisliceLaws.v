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

End MultisliceLaws.

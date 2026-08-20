Require Import List.

Import ListNotations.

Section FlattenLaws.

Variable A : Type.

Definition flatten (xss : list (list A)) : list A :=
  concat xss.

Lemma flatten_empty :
  flatten [] = [].
Proof.
  reflexivity.
Qed.

Lemma flatten_singleton :
  forall xs : list A,
    flatten [xs] = xs.
Proof.
  intro xs.
  simpl.
  rewrite app_nil_r.
  reflexivity.
Qed.

Lemma flatten_app :
  forall left right : list (list A),
    flatten (left ++ right) = flatten left ++ flatten right.
Proof.
  intros left right.
  unfold flatten.
  apply concat_app.
Qed.

End FlattenLaws.

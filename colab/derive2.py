"""derive.py + more natural-product edits (carboxyl, hydration/dehydration, methylenedioxy, alcohol->acid, epoxide,
further acyl/sugar removals). Same API: derive(parent_smiles, delta, tol, max_steps)."""
import itertools
from rdkit.Chem import AllChem
import derive as base
from derive import fm, _apply

EXTRA = [
    ('+CO2', '[cH:1]>>[c:1]C(=O)O', fm(C=1, O=2)),
    ('-CO2', '[#6:1]-C(=O)-[OH1]>>[#6:1]', -fm(C=1, O=2)),
    ('+H2O', '[C;!a:1]=[C;!a:2]>>[C:1]-[C:2]O', fm(H=2, O=1)),
    ('-H2O', '[CX4;!H0:1]-[CX4:2]-[OH1]>>[C:1]=[C:2]', -fm(H=2, O=1)),
    ('OCH2O', '[c:1](-[OH1:3]):[c:2]-[OH1:4]>>[c:1]1:[c:2]-[O:4]-C-[O:3]-1', fm(C=1)),
    ('de-OCH2O', '[c:1]1:[c:2]-[O:4]-[CH2]-[O:3]-1>>[c:1](-[O:3]):[c:2]-[O:4]', -fm(C=1)),
    ('CH2OH>COOH', '[CX4H2:1]-[OH1:2]>>[C:1](=O)-[O:2]', fm(O=1) - fm(H=2)),
    ('COOH>CH2OH', '[CX3:1](=O)-[OH1:2]>>[CH2:1]-[O:2]', fm(H=2) - fm(O=1)),
    ('CH3>CH2OH', '[CH3:1]-[#6:2]>>[CH2:1](O)-[#6:2]', fm(O=1)),
    ('epox', '[C;!a:1]=[C;!a:2]>>[C:1]1-[C:2]-O-1', fm(O=1)),
    ('de-O-GlcA', '[O:1]-[CH1;R]1-[O;R]-[CH1;R](-C(=O)[OH1])-[CH1;R](-[OH1])-[CH1;R](-[OH1])-[CH1;R]1-[OH1]>>[O:1]', -fm(C=6, H=8, O=6)),
    ('de-O-Mal', '[O:1]-C(=O)-[CH2]-C(=O)-[OH1]>>[O:1]', -fm(C=3, H=2, O=3)),
    ('de-fer', '[O:1]-C(=O)-C=C-c1ccc([OH1])c(O[CH3])c1>>[O:1]', -fm(C=10, H=8, O=3)),
    ('de-gall', '[O:1]-C(=O)-c1cc([OH1])c([OH1])c([OH1])c1>>[O:1]', -fm(C=7, H=4, O=4)),
    ('de-O-prenylO', '[O:1]-[CH2]-[CH1]=[C](-[CH3])-[CH3]>>[O:1]', -fm(C=5, H=8)),
    ('O-HexOMe', '[OX2H1:1]>>[O:1]C1OC(COC)C(O)C(O)C1O', fm(C=7, H=12, O=5)),
    ('N-Ac', '[NX3;!H0;!$(N-C=O):1]>>[N:1]C(C)=O', fm(C=2, H=2, O=1)),
    ('de-N-Ac', '[N:1]-C(=O)-[CH3]>>[N:1]', -fm(C=2, H=2, O=1)),
]
TR = base.TR + [(n, AllChem.ReactionFromSmarts(s), d) for n, s, d in EXTRA]


def plans(delta, tol, max_steps=2):
    out = []
    for k in range(1, max_steps + 1):
        for combo in itertools.combinations_with_replacement(range(len(TR)), k):
            if abs(sum(TR[i][2] for i in combo) - delta) <= tol:
                out.append(combo)
    return out


def derive(parent_smiles, delta, tol=0.004, max_steps=2, max_per_parent=300):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(parent_smiles)
    if mol is None:
        return {}
    res = {}
    for combo in plans(delta, tol, max_steps):
        for perm in set(itertools.permutations(combo)):
            cur = {Chem.MolToSmiles(mol): mol}
            for i in perm:
                nxt = {}
                for m in cur.values():
                    nxt.update(_apply(m, TR[i][1]))
                    if len(nxt) > max_per_parent:
                        break
                cur = nxt
                if not cur:
                    break
            for s in cur:
                res.setdefault(s, '+'.join(TR[i][0] for i in perm))
            if len(res) > max_per_parent:
                return res
    return res

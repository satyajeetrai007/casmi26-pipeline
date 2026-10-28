"""Natural-product derivative generator (class-3 candidates).

derive(parent_smiles, delta, tol) applies one or two biosynthetic-style edits (methylation, hydroxylation,
glycosylation, acylation, (de)hydrogenation and their reverses) whose summed exact mass change matches delta,
at every matching site, and returns the product SMILES. Products are checked by exact mass by the caller.
"""
import itertools
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
RDLogger.DisableLog('rdApp.*')

M = dict(C=12.0, H=1.00782503207, O=15.99491461956, N=14.0030740048, S=31.97207100)
def fm(**k): return sum(M[e] * n for e, n in k.items())

HEX = 'C1OC(CO)C(O)C(O)C1O'; DHEX = 'C1OC(C)C(O)C(O)C1O'; PEN = 'C1OCC(O)C(O)C1O'; GLA = 'C1OC(C(=O)O)C(O)C(O)C1O'
# name, reaction SMARTS, exact mass change
FWD = [
    ('O-Me', '[OX2H1:1]>>[O:1]C', fm(C=1, H=2)),
    ('N-Me', '[NX3;!H0;!$(N=*):1]>>[N:1]C', fm(C=1, H=2)),
    ('C-Me', '[cH:1]>>[c:1]C', fm(C=1, H=2)),
    ('ar-OH', '[cH:1]>>[c:1]O', fm(O=1)),
    ('al-OH', '[CX4;!H0:1]>>[C:1]O', fm(O=1)),
    ('O-Hex', '[OX2H1:1]>>[O:1]' + HEX, fm(C=6, H=10, O=5)),
    ('C-Hex', '[cH:1]>>[c:1]' + HEX, fm(C=6, H=10, O=5)),
    ('O-dHex', '[OX2H1:1]>>[O:1]' + DHEX, fm(C=6, H=10, O=4)),
    ('O-Pen', '[OX2H1:1]>>[O:1]' + PEN, fm(C=5, H=8, O=4)),
    ('O-GlcA', '[OX2H1:1]>>[O:1]' + GLA, fm(C=6, H=8, O=6)),
    ('O-Ac', '[OX2H1:1]>>[O:1]C(C)=O', fm(C=2, H=2, O=1)),
    ('O-Mal', '[OX2H1:1]>>[O:1]C(=O)CC(=O)O', fm(C=3, H=2, O=3)),
    ('O-SO3', '[OX2H1:1]>>[O:1]S(=O)(=O)O', fm(S=1, O=3)),
    ('C-prenyl', '[cH:1]>>[c:1]CC=C(C)C', fm(C=5, H=8)),
    ('O-prenyl', '[OX2H1:1]>>[O:1]CC=C(C)C', fm(C=5, H=8)),
    ('O-caff', '[OX2H1:1]>>[O:1]C(=O)C=Cc1ccc(O)c(O)c1', fm(C=9, H=6, O=3)),
    ('O-coum', '[OX2H1:1]>>[O:1]C(=O)C=Cc1ccc(O)cc1', fm(C=9, H=6, O=2)),
    ('O-fer', '[OX2H1:1]>>[O:1]C(=O)C=Cc1ccc(O)c(OC)c1', fm(C=10, H=8, O=3)),
    ('O-gall', '[OX2H1:1]>>[O:1]C(=O)c1cc(O)c(O)c(O)c1', fm(C=7, H=4, O=4)),
    ('+H2', '[C;!a:1]=[C;!a:2]>>[C:1]-[C:2]', fm(H=2)),
    ('C=O>CH-OH', '[C;!a:1]=[O:2]>>[C:1]-[O:2]', fm(H=2)),
]
REV = [
    ('de-O-Me', '[O:1]-[CH3]>>[O:1]', -fm(C=1, H=2)),
    ('de-N-Me', '[N:1]-[CH3]>>[N:1]', -fm(C=1, H=2)),
    ('de-C-Me', '[c:1]-[CH3]>>[c:1]', -fm(C=1, H=2)),
    ('de-ar-OH', '[c:1]-[OH1]>>[c:1]', -fm(O=1)),
    ('de-al-OH', '[CX4:1]-[OH1]>>[C:1]', -fm(O=1)),
    ('de-O-Hex', '[O:1]-[CH1;R]1-[O;R]-[CH1;R](-[CH2]-[OH1])-[CH1;R](-[OH1])-[CH1;R](-[OH1])-[CH1;R]1-[OH1]>>[O:1]', -fm(C=6, H=10, O=5)),
    ('de-C-Hex', '[c:1]-[CH1;R]1-[O;R]-[CH1;R](-[CH2]-[OH1])-[CH1;R](-[OH1])-[CH1;R](-[OH1])-[CH1;R]1-[OH1]>>[c:1]', -fm(C=6, H=10, O=5)),
    ('de-O-dHex', '[O:1]-[CH1;R]1-[O;R]-[CH1;R](-[CH3])-[CH1;R](-[OH1])-[CH1;R](-[OH1])-[CH1;R]1-[OH1]>>[O:1]', -fm(C=6, H=10, O=4)),
    ('de-O-Pen', '[O:1]-[CH1;R]1-[O;R]-[CH2;R]-[CH1;R](-[OH1])-[CH1;R](-[OH1])-[CH1;R]1-[OH1]>>[O:1]', -fm(C=5, H=8, O=4)),
    ('de-O-Ac', '[O:1]-C(=O)-[CH3]>>[O:1]', -fm(C=2, H=2, O=1)),
    ('de-O-SO3', '[O:1]-S(=O)(=O)-[OH1]>>[O:1]', -fm(S=1, O=3)),
    ('de-prenyl', '[c,O:1]-[CH2]-[CH1]=[C](-[CH3])-[CH3]>>[*:1]', -fm(C=5, H=8)),
    ('de-caff', '[O:1]-C(=O)-C=C-c1ccc([OH1])c([OH1])c1>>[O:1]', -fm(C=9, H=6, O=3)),
    ('de-coum', '[O:1]-C(=O)-C=C-c1ccc([OH1])cc1>>[O:1]', -fm(C=9, H=6, O=2)),
    ('-H2', '[C;!a;!H0:1]-[C;!a;!H0:2]>>[C:1]=[C:2]', -fm(H=2)),
    ('CH-OH>C=O', '[C;!a;!H0:1]-[OH1:2]>>[C:1]=[O:2]', -fm(H=2)),
]
TR = [(n, AllChem.ReactionFromSmarts(s), d) for n, s, d in FWD + REV]


def _apply(mol, rxn, cap=200):
    out = {}
    try:
        res = rxn.RunReactants((mol,), maxProducts=cap)
    except Exception:
        return out
    for prods in res:
        p = prods[0]
        try:
            Chem.SanitizeMol(p)
            s = Chem.MolToSmiles(p)
            m = Chem.MolFromSmiles(s)
        except Exception:
            continue
        if m is not None and s not in out:
            out[s] = m
    return out


def plans(delta, tol, max_steps=2):
    """Transform index sequences whose exact mass change sums to delta within tol."""
    out = []
    for k in range(1, max_steps + 1):
        for combo in itertools.combinations_with_replacement(range(len(TR)), k):
            if abs(sum(TR[i][2] for i in combo) - delta) <= tol:
                out.append(combo)
    return out


def derive(parent_smiles, delta, tol=0.004, max_steps=2, max_per_parent=300):
    """Products of parent whose mass is parent + delta (within tol). Returns {smiles: plan names}."""
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


def exact_mass(smi):
    m = Chem.MolFromSmiles(smi)
    return None if m is None else Descriptors.ExactMolWt(m)

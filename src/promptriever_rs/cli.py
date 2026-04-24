from __future__ import annotations

import argparse

from promptriever_rs.data.instruction_dataset import assemble_instruction_dataset
from promptriever_rs.data.sberquad import build_sberquad_records
from promptriever_rs.evaluation.mfollowir import evaluate_mfollowir
from promptriever_rs.evaluation.mteb_eval import evaluate_mteb
from promptriever_rs.generation.groq_llama import generate_negative_instructions
from promptriever_rs.training.train import fit


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="promptriever-rs")
    subparsers = parser.add_subparsers(dest="group", required=True)

    data_parser = subparsers.add_parser("data")
    data_subparsers = data_parser.add_subparsers(dest="command", required=True)
    build_sberquad = data_subparsers.add_parser("build-sberquad")
    build_sberquad.add_argument("--config", required=True)
    assemble = data_subparsers.add_parser("assemble-training-set")
    assemble.add_argument("--config", required=True)

    generation_parser = subparsers.add_parser("generation")
    generation_subparsers = generation_parser.add_subparsers(dest="command", required=True)
    generate = generation_subparsers.add_parser("generate-negatives")
    generate.add_argument("--config", required=True)

    train_parser = subparsers.add_parser("train")
    train_subparsers = train_parser.add_subparsers(dest="command", required=True)
    fit_parser = train_subparsers.add_parser("fit")
    fit_parser.add_argument("--config", required=True)

    eval_parser = subparsers.add_parser("eval")
    eval_subparsers = eval_parser.add_subparsers(dest="command", required=True)
    rumteb = eval_subparsers.add_parser("rumteb")
    rumteb.add_argument("--config", required=True)
    mfollowir = eval_subparsers.add_parser("mfollowir")
    mfollowir.add_argument("--config", required=True)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.group == "data" and args.command == "build-sberquad":
        path = build_sberquad_records(args.config)
        print(path)
        return
    if args.group == "data" and args.command == "assemble-training-set":
        path = assemble_instruction_dataset(args.config)
        print(path)
        return
    if args.group == "generation" and args.command == "generate-negatives":
        path = generate_negative_instructions(args.config)
        print(path)
        return
    if args.group == "train" and args.command == "fit":
        path = fit(args.config)
        print(path)
        return
    if args.group == "eval" and args.command == "rumteb":
        path = evaluate_mteb(args.config)
        print(path)
        return
    if args.group == "eval" and args.command == "mfollowir":
        path = evaluate_mfollowir(args.config)
        print(path)
        return

    parser.error("Unsupported command")

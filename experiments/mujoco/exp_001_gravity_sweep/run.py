"""重力条件ごとの落下挙動を比較するMuJoCo実験。

地球・火星・月などの重力条件をYAMLファイルから読み込み、
同一の球体モデルを落下させます。

各条件について、以下の指標をCSVへ保存します。

- 最初に床へ接触した時刻
- 接触直前の鉛直速度
- シミュレーション中の最大鉛直速度

実行例:
    python run.py
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import mujoco
import yaml


# このrun.pyが存在するディレクトリ。
# 実行時のカレントディレクトリに依存せず、
# config.yamlやresults.csvを正しく参照するために使用します。
ROOT = Path(__file__).resolve().parent

# 実験条件を記述したYAMLファイル。
CONFIG_PATH = ROOT / "config.yaml"

# 実験結果を書き出すCSVファイル。
RESULTS_PATH = ROOT / "results.csv"


# MuJoCoへ渡すXMLモデルのテンプレートです。
#
# Pythonのstr.format()を使って、以下の値を実験ごとに差し替えます。
#
# - timestep: シミュレーションの時間刻み
# - gravity: 重力加速度
# - initial_height: 球の初期高度
MODEL_TEMPLATE = """
<mujoco model="gravity_sweep">
    <option
        timestep="{timestep}"
        gravity="0 0 -{gravity}"
    />

    <worldbody>
        <!-- 無限平面として扱われる床 -->
        <geom
            name="floor"
            type="plane"
            size="5 5 0.1"
        />

        <!-- 落下対象となる球体 -->
        <body
            name="ball"
            pos="0 0 {initial_height}"
        >
            <!--
            freejointを付けることで、球体がワールド内を
            3次元的に自由移動・自由回転できるようになります。
            -->
            <freejoint/>

            <geom
                name="ball_geom"
                type="sphere"
                size="0.15"
                mass="1.0"
            />
        </body>
    </worldbody>
</mujoco>
"""


def load_config() -> dict[str, Any]:
    """YAMLファイルから実験設定を読み込む。

    Returns:
        config.yamlの内容を格納した辞書。

    Raises:
        FileNotFoundError:
            config.yamlが存在しない場合。
        yaml.YAMLError:
            YAMLの構文が不正な場合。
        ValueError:
            YAMLファイルが空の場合。
    """
    # UTF-8でYAMLファイルを開きます。
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # 空ファイルの場合、safe_load()はNoneを返します。
    if config is None:
        raise ValueError(f"設定ファイルが空です: {CONFIG_PATH}")

    return config


def run_condition(
    name: str,
    gravity: float,
    timestep: float,
    duration_seconds: float,
    initial_height: float,
) -> dict[str, float | str]:
    """1つの重力条件で落下シミュレーションを実行する。

    Args:
        name:
            条件名。例としてearth、mars、moonなどを指定します。
        gravity:
            重力加速度の大きさ。単位はm/s^2です。
            XML内ではZ軸負方向の重力として設定されます。
        timestep:
            シミュレーション1ステップあたりの時間。
            単位は秒です。
        duration_seconds:
            シミュレーションを継続する時間。
            単位は秒です。
        initial_height:
            球体中心の初期高度。
            単位はメートルです。

    Returns:
        実験条件と計測結果を格納した辞書。

        次の値を含みます。

        - condition: 条件名
        - gravity: 重力加速度
        - impact_time: 最初の接触時刻
        - impact_velocity: 接触直前の鉛直速度
        - maximum_speed: 最大鉛直速度

    Raises:
        ValueError:
            MuJoCoモデル内にballという名前のbodyが存在しない場合。
    """
    # 実験条件をXMLテンプレートへ埋め込みます。
    xml = MODEL_TEMPLATE.format(
        timestep=timestep,
        gravity=gravity,
        initial_height=initial_height,
    )

    # XML文字列からMuJoCoのモデルを生成します。
    #
    # MjModel:
    #   質量、形状、関節、重力、時間刻みなど、
    #   シミュレーション中に基本的に変化しない情報を保持します。
    model = mujoco.MjModel.from_xml_string(xml)

    # シミュレーション中に変化する状態を格納します。
    #
    # MjData:
    #   時刻、位置、速度、接触情報などを保持します。
    data = mujoco.MjData(model)

    # 名前が"ball"であるbodyの数値IDを取得します。
    #
    # MuJoCo内部では、各bodyやgeomを整数IDで管理しています。
    ball_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "ball",
    )

    if ball_body_id < 0:
        raise ValueError("MuJoCoモデル内にbody 'ball' がありません。")

    # シミュレーション中に観測された最大速度。
    maximum_speed = 0.0

    # まだ接触していないため、初期値はNoneにします。
    impact_time: float | None = None
    impact_velocity: float | None = None

    # 指定したシミュレーション時間まで繰り返します。
    while data.time < duration_seconds:
        # cvelは各bodyの6次元速度を表します。
        #
        # 先頭3要素:
        #   角速度
        #
        # 後半3要素:
        #   並進速度
        #
        # インデックス5は並進速度のZ成分です。
        # 下向きに落下している間は負の値になります。
        vertical_velocity = float(data.cvel[ball_body_id][5])

        # 最大速度では方向を無視するため、絶対値を使います。
        speed = abs(vertical_velocity)
        maximum_speed = max(maximum_speed, speed)

        # 物理シミュレーションを1ステップ進めます。
        mujoco.mj_step(model, data)

        # nconは現在検出されている接触点の数です。
        #
        # 今回は球と床しか動的な接触候補がないため、
        # nconが1以上になった最初の時点を衝突として扱います。
        if impact_time is None and data.ncon > 0:
            impact_time = float(data.time)

            # mj_step()直前に取得した速度を、
            # 接触直前付近の速度として記録します。
            impact_velocity = vertical_velocity

    # 接触が発生しなかった場合は空文字を出力します。
    #
    # floatと空文字が混在するため、
    # 戻り値の型はfloat | strになっています。
    return {
        "condition": name,
        "gravity": gravity,
        "impact_time": impact_time if impact_time is not None else "",
        "impact_velocity": (
            impact_velocity if impact_velocity is not None else ""
        ),
        "maximum_speed": maximum_speed,
    }


def write_results(rows: list[dict[str, float | str]]) -> None:
    """実験結果をCSVファイルへ書き込む。

    Args:
        rows:
            条件ごとの実験結果を格納した辞書のリスト。

    Raises:
        OSError:
            CSVファイルの作成や書き込みに失敗した場合。
    """
    # CSVの列順を明示します。
    fieldnames = [
        "condition",
        "gravity",
        "impact_time",
        "impact_velocity",
        "maximum_speed",
    ]

    # newline=""は、WindowsでCSVへ余分な空行が入ることを防ぎます。
    with RESULTS_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        # 1行目に列名を書き込みます。
        writer.writeheader()

        # 条件ごとの結果を書き込みます。
        writer.writerows(rows)


def main() -> None:
    """設定の読み込みから結果保存まで、実験全体を実行する。"""
    # YAMLから実験条件を読み込みます。
    config = load_config()

    # よく使う階層を変数へ取り出します。
    simulation = config["simulation"]
    gravity_conditions = config["conditions"]["gravity"]

    # 全条件の結果を格納します。
    rows: list[dict[str, float | str]] = []

    # earth、mars、moonなどを順番に実行します。
    for name, gravity in gravity_conditions.items():
        print(f"Running condition: {name}")

        result = run_condition(
            name=name,
            gravity=float(gravity),
            timestep=float(simulation["timestep"]),
            duration_seconds=float(
                simulation["duration_seconds"]
            ),
            initial_height=float(
                simulation["initial_height"]
            ),
        )

        rows.append(result)
        print(result)

    # すべての条件が終了した後にCSVへ保存します。
    write_results(rows)

    print(f"Saved results: {RESULTS_PATH}")


# run.pyを直接実行した場合だけmain()を呼び出します。
#
# 別のPythonファイルからimportされた場合には、
# main()が勝手に実行されません。
if __name__ == "__main__":
    main()
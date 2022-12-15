module.exports = (sequelize, DataTypes) => {
    return sequelize.define('TRANSACTION', {
      id: {
        type: DataTypes.INTEGER,
        primaryKey: true,
        autoIncrement: true
      },
      montant_a_recevoir: {
        type: DataTypes.FLOAT,
        
      },
      montantTotal: {
        type: DataTypes.FLOAT
      },
      status: {
        type: DataTypes.STRING,
      },
      paysDest: {
        type: DataTypes.STRING,
      },
      paysOrigine: {
        type: DataTypes.STRING,
      },
      DeviceDest: {
        type: DataTypes.STRING,
       },
       DeviceOrigine: {
        type: DataTypes.STRING,
       },
      frais:{
        type:DataTypes.FLOAT
      },
      recepteurid:{
        type:DataTypes.INTEGER
      },
      

    })
  }